import argparse
import copy
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, save_checkpoint
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from methods.common.coco_pool import build_coco_subset, image_ids, read_coco_json, write_coco_json
from methods.common.selection import deterministic_random_sample
from mmdet.alod.datasets import *
from mmdet.alod.models import *
from tools.common.mmcv_compat import patch_yapf_verify_arg


def parse_args():
    parser = argparse.ArgumentParser(description='Train MIAL detector')
    parser.add_argument('config', help='MIAL train config file path')
    parser.add_argument('--work-dir', required=True, help='Round work directory')
    parser.add_argument('--labeled-ann', required=True, help='Current labeled COCO pool JSON')
    parser.add_argument('--unlabeled-ann', required=True, help='Current unlabeled COCO pool JSON')
    parser.add_argument('--uncertainty-out', required=True, help='Output MIAL uncertainty JSON')
    parser.add_argument('--round-index', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.launcher != 'none':
        raise NotImplementedError(
            'tools/internal/train_mial_detector.py currently supports --launcher none only')
    if args.gpus != 1:
        raise NotImplementedError(
            'tools/internal/train_mial_detector.py currently supports one GPU per seed')
    return args


def _prepare_cfg(args):
    patch_yapf_verify_arg()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir
    cfg.gpu_ids = range(args.gpus)
    cfg.seed = int(args.seed)
    cfg.data.train.ann_file = args.labeled_ann
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    return cfg


def _build_pool_dataset_cfg(cfg, ann_file, pipeline, filter_empty_gt=True):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    dataset_cfg.ann_file = str(ann_file)
    dataset_cfg.pipeline = copy.deepcopy(pipeline)
    dataset_cfg.filter_empty_gt = bool(filter_empty_gt)
    dataset_cfg.test_mode = False
    return dataset_cfg


def _build_infer_dataset_cfg(cfg, ann_file):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    dataset_cfg.ann_file = str(ann_file)
    dataset_cfg.pipeline = copy.deepcopy(cfg.test_pipeline)
    dataset_cfg.test_mode = True
    return dataset_cfg


def _sample_unlabeled_training_pool(unlabeled_ann, labeled_ann, output_path, seed):
    unlabeled = read_coco_json(Path(unlabeled_ann))
    labeled = read_coco_json(Path(labeled_ann))
    unlabeled_ids = image_ids(unlabeled)
    target_count = min(len(unlabeled_ids), len(image_ids(labeled)))
    sampled = deterministic_random_sample(unlabeled_ids, target_count, seed=seed)
    write_coco_json(
        build_coco_subset(unlabeled, sampled, include_annotations=False),
        Path(output_path),
    )
    return output_path


def _build_loader(cfg, dataset, shuffle):
    return build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        shuffle=shuffle,
        seed=cfg.seed,
        runner_type='EpochBasedRunner',
        persistent_workers=cfg.data.get('persistent_workers', False))


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _set_mial_phase(model, phase, unlabeled=False):
    head = _unwrap(model).bbox_head
    if not hasattr(head, 'set_mial_phase'):
        raise TypeError('MIAL train requires a bbox_head with set_mial_phase')
    head.set_mial_phase(phase, unlabeled=unlabeled)


def _set_requires_grad(model, phase):
    module = _unwrap(model)
    for name, param in module.named_parameters():
        if phase == 'det':
            param.requires_grad = True
        elif phase == 'min':
            param.requires_grad = not (
                name.startswith('bbox_head.f_1_')
                or name.startswith('bbox_head.f_2_'))
        elif phase == 'max':
            param.requires_grad = (
                name.startswith('bbox_head.f_1_')
                or name.startswith('bbox_head.f_2_'))
        else:
            raise ValueError('Unsupported MIAL phase: %s' % phase)


def _planned_phase_epochs(cfg):
    schedule = cfg.get('mial_phase_schedule', {})
    outer_loops = int(schedule.get('outer_loops', 2))
    epoch_ratio = list(schedule.get('epoch_ratio', [3, 1]))
    repeat_factor = int(schedule.get('repeat_factor', 2))
    det_epochs = int(epoch_ratio[0]) * repeat_factor
    wave_epochs = int(epoch_ratio[1]) * repeat_factor
    phases = []
    for outer_idx in range(outer_loops):
        if outer_idx == 0:
            phases.append(('det', det_epochs))
        phases.extend([
            ('min', wave_epochs),
            ('max', wave_epochs),
            ('det', det_epochs),
        ])
    return phases


def _total_iterations(phases, labeled_len, unlabeled_len):
    total = 0
    for phase, epochs in phases:
        if phase == 'det':
            total += epochs * labeled_len
        else:
            total += epochs * (labeled_len + min(labeled_len, unlabeled_len))
    return max(total, 1)


def _base_lr(cfg):
    return float(cfg.optimizer.get('lr', 0.002))


def _lr_gamma(cfg):
    return float(cfg.lr_config.get('gamma', 0.1))


def _scheduled_lr(cfg, global_epoch, global_iter):
    lr = _base_lr(cfg)
    steps = cfg.lr_config.get('step', [])
    if isinstance(steps, int):
        exp = global_epoch // int(steps)
    else:
        exp = sum(1 for step in steps if global_epoch >= int(step))
    lr *= _lr_gamma(cfg) ** exp

    warmup = cfg.lr_config.get('warmup')
    warmup_iters = int(cfg.lr_config.get('warmup_iters', 0))
    if warmup == 'linear' and global_iter < warmup_iters:
        warmup_ratio = float(cfg.lr_config.get('warmup_ratio', 0.001))
        alpha = global_iter / max(warmup_iters, 1)
        lr *= warmup_ratio + alpha * (1.0 - warmup_ratio)
    return lr


def _set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def _grad_norm(parameters):
    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().data.norm(2))
    if not norms:
        return 0.0
    total = torch.norm(torch.stack(norms), 2)
    return float(total.item())


def _optimizer_step(cfg, model, optimizer):
    grad_cfg = cfg.optimizer_config.get('grad_clip') if cfg.get('optimizer_config') else None
    if grad_cfg:
        torch.nn.utils.clip_grad_norm_(
            [p for p in _unwrap(model).parameters() if p.requires_grad and p.grad is not None],
            max_norm=float(grad_cfg.get('max_norm', 35)),
            norm_type=float(grad_cfg.get('norm_type', 2)))
    optimizer.step()


def _format_eta(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def _log_interval(cfg):
    log_config = cfg.get('log_config', {})
    return max(int(log_config.get('interval', 50)), 1)


def _should_print_progress(cfg, global_state, total_iters):
    current = int(global_state['iter'])
    return current >= total_iters or current % _log_interval(cfg) == 0


def _train_one_batch(model, optimizer, data, phase, unlabeled):
    _set_mial_phase(model, phase, unlabeled=unlabeled)
    optimizer.zero_grad()
    outputs = model.train_step(data, optimizer)
    loss = outputs['loss']
    loss.backward()
    grad_norm = _grad_norm(_unwrap(model).parameters())
    return outputs, grad_norm


def _train_phase(model, cfg, phase, epochs, labeled_loader, unlabeled_loader,
                 global_state, total_iters):
    _set_requires_grad(model, phase)
    optimizer = build_optimizer(model, cfg.optimizer)
    unlabeled_iter = None
    for _ in range(epochs):
        global_state['epoch'] += 1
        if phase != 'det':
            unlabeled_iter = iter(unlabeled_loader)
        for data in labeled_loader:
            global_state['iter'] += 1
            lr = _scheduled_lr(cfg, global_state['epoch'] - 1, global_state['iter'] - 1)
            _set_optimizer_lr(optimizer, lr)
            outputs, grad_norm = _train_one_batch(
                model, optimizer, data, phase, unlabeled=False)
            _optimizer_step(cfg, model, optimizer)
            if _should_print_progress(cfg, global_state, total_iters):
                _print_progress(
                    global_state, total_iters, phase, lr, outputs, grad_norm)
            if phase == 'det':
                continue
            try:
                unlabeled_data = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_data = next(unlabeled_iter)
            global_state['iter'] += 1
            lr = _scheduled_lr(cfg, global_state['epoch'] - 1, global_state['iter'] - 1)
            _set_optimizer_lr(optimizer, lr)
            outputs, grad_norm = _train_one_batch(
                model, optimizer, unlabeled_data, phase, unlabeled=True)
            _optimizer_step(cfg, model, optimizer)
            if _should_print_progress(cfg, global_state, total_iters):
                _print_progress(
                    global_state, total_iters, phase + '_unlabeled', lr,
                    outputs, grad_norm)


def _print_progress(global_state, total_iters, phase, lr, outputs, grad_norm):
    current = min(global_state['iter'], total_iters)
    elapsed = time.time() - global_state['start_time']
    speed = elapsed / max(current, 1)
    eta = _format_eta((total_iters - current) * speed)
    log_vars = outputs.get('log_vars', {})
    loss = float(outputs['loss'].detach().cpu())
    if 'loss' in log_vars:
        loss = float(log_vars['loss'])
    parts = [
        'Iter [%d/%d]' % (current, total_iters),
        'epoch: %d' % global_state['epoch'],
        'iter: %d' % current,
        'phase: %s' % phase,
        'lr: %.6g' % lr,
        'loss: %.4f' % loss,
        'grad_norm: %.4f' % grad_norm,
        'eta: %s' % eta,
    ]
    for key in sorted(log_vars):
        if key == 'loss':
            continue
        value = log_vars[key]
        if isinstance(value, (int, float)):
            parts.append('%s: %.4f' % (key, float(value)))
    print(', '.join(parts), flush=True)


def _write_uncertainty(model, cfg, unlabeled_ann, output_path, round_index):
    dataset = build_dataset(_build_infer_dataset_cfg(cfg, unlabeled_ann))
    data_loader = _build_loader(cfg, dataset, shuffle=False)
    module = _unwrap(model)
    module.eval()
    topk = int(cfg.get('mial_topk', 10000))
    records = []
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            img = data['img'][0].cuda()
            img_metas = data['img_metas'][0].data[0]
            feats = module.extract_feat(img)
            uncertainty_records = module.bbox_head.image_uncertainty(
                feats, topk=topk)
            for local_idx, item in enumerate(uncertainty_records):
                image_index = idx * cfg.data.samples_per_gpu + local_idx
                if image_index >= len(dataset.img_ids):
                    continue
                image_id = dataset.img_ids[image_index]
                score = float(item['score'].detach().cpu())
                records.append({
                    'image_id': image_id,
                    'score': score,
                    'components': {
                        'topk_mean_discrepancy': score,
                        'instance_count': int(item['instance_count']),
                        'effective_topk': int(item['effective_topk']),
                    },
                    'metadata': {
                        'filename': img_metas[local_idx].get('filename')
                        if local_idx < len(img_metas) else None,
                    },
                })
            if (idx + 1) % 50 == 0 or idx + 1 == len(data_loader):
                print('%d / %d' % (idx + 1, len(data_loader)), flush=True)
    payload = {
        'method': 'mial',
        'stage': 'instance_discrepancy',
        'round_index': int(round_index),
        'topk': topk,
        'record_count': len(records),
        'records': records,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    cfg = _prepare_cfg(args)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    logger = get_root_logger(log_level=cfg.log_level)

    env_info = '\n'.join('%s: %s' % (key, value) for key, value in collect_env().items())
    logger.info('Environment info:\n%s', env_info)
    logger.info('Distributed training: False')
    logger.info('Config:\n%s', cfg.pretty_text)
    set_random_seed(args.seed, deterministic=args.deterministic)

    unlabeled_train_ann = Path(cfg.work_dir) / 'mial_unlabeled_train.json'
    _sample_unlabeled_training_pool(
        args.unlabeled_ann,
        args.labeled_ann,
        unlabeled_train_ann,
        seed=args.seed + args.round_index)

    labeled_dataset = build_dataset(
        _build_pool_dataset_cfg(
            cfg, args.labeled_ann, cfg.train_pipeline, filter_empty_gt=True))
    unlabeled_dataset = build_dataset(
        _build_pool_dataset_cfg(
            cfg, unlabeled_train_ann, cfg.train_pipeline,
            filter_empty_gt=False))
    labeled_loader = _build_loader(cfg, labeled_dataset, shuffle=True)
    unlabeled_loader = _build_loader(cfg, unlabeled_dataset, shuffle=True)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.CLASSES = labeled_dataset.CLASSES
    meta = dict(
        mmdet_version=__version__ + get_git_hash()[:7],
        CLASSES=labeled_dataset.CLASSES,
        seed=args.seed,
        config=cfg.pretty_text)
    model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    phases = _planned_phase_epochs(cfg)
    total_iters = _total_iterations(phases, len(labeled_loader), len(unlabeled_loader))
    global_state = {'epoch': 0, 'iter': 0, 'start_time': time.time()}
    for phase, epochs in phases:
        _train_phase(
            model,
            cfg,
            phase,
            epochs,
            labeled_loader,
            unlabeled_loader,
            global_state,
            total_iters)

    checkpoint_path = Path(cfg.work_dir) / 'latest.pth'
    save_checkpoint(
        _unwrap(model),
        str(checkpoint_path),
        optimizer=None,
        meta=meta)
    _write_uncertainty(model, cfg, args.unlabeled_ann, args.uncertainty_out,
                       args.round_index)


if __name__ == '__main__':
    main()
