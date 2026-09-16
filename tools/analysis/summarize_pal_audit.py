"""Cross-check saved PAL candidate scores and summarize real audit outputs."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.pal_error_audit import (
    DETECTION_TYPES, ERRORS, auc, read_json, write_csv, write_json,
)


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def predictor_details(source, output):
    checks, class_types, calibration, strata, visibility = [], [], [], [], []
    categories = read_json('data/VOC0712/annotations/trainval_0712.json')['categories']
    names = {c['id']: c['name'] for c in categories}
    for index in range(1, 7):
        data = np.load(output / ('round_%02d' % index) / 'predictor_detections.npz')
        image, category = data['image_id'], data['category_id']
        entropy, q_fp, score = data['entropy'], data['q_fp'], data['score']
        label = data['error_type']
        valid = label != 'Ignored'
        best = {}
        for j in range(len(image)):
            key = (int(image[j]), int(category[j]))
            current = best.get(key)
            if current is None or (entropy[j], score[j]) > (entropy[current], score[current]):
                best[key] = j
        candidates = read_json(source / ('round_%02d' % index) / 'pal_candidates.json')['candidates']
        differences = []
        for candidate in candidates:
            j = best[(candidate['image_id'], candidate['category_id'])]
            components = candidate['components']
            differences.extend((abs(entropy[j] - components['lius_score']),
                                abs(1 - q_fp[j] - components['tp_probability'])))
        maximum = max(differences)
        if maximum > 1e-9:
            raise ValueError('Saved PAL candidate scores not reproduced in round %d: %g' % (index, maximum))
        checks.append(dict(round=index, candidates=len(candidates), max_score_difference=maximum))
        fp = label != 'TP'
        calibration.append(dict(round=index, mean_q_fp=float(q_fp[valid].mean()),
                                observed_fp=float(fp[valid].mean()),
                                signed_calibration_bias=float(q_fp[valid].mean() - fp[valid].mean())))
        for cid, name in names.items():
            class_valid = valid & (category == cid)
            local_order = np.flatnonzero(class_valid)[np.argsort(-entropy[class_valid], kind='stable')]
            top = np.zeros(len(label), dtype=bool)
            top[local_order[:int(np.ceil(len(local_order) * .1))]] = True
            for kind in DETECTION_TYPES:
                subtype = (label == kind) & (category == cid)
                pair = ((label == 'TP') | (label == kind)) & (category == cid)
                if not subtype.any():
                    continue
                class_types.append(dict(round=index, category=name, type=kind,
                    n=int(subtype.sum()), q_fp_mean=float(q_fp[subtype].mean()),
                    entropy_mean=float(entropy[subtype].mean()),
                    auc_vs_tp=auc((label[pair] == kind).astype(int), q_fp[pair]) if kind != 'TP' else None,
                    score_auc_vs_tp=auc((label[pair] == kind).astype(int), 1 - score[pair]) if kind != 'TP' else None,
                    within_class_top10_coverage=float(top[subtype].mean())))
        for low, high in ((0, .1), (.1, .4), (.4, .6), (.6, .9), (.9, 1.0000001)):
            take = valid & (q_fp >= low) & (q_fp < high)
            strata.append(dict(round=index, q_fp_low=low, q_fp_high=min(high, 1),
                               n=int(take.sum()), fp_fraction=float(fp[take].mean()) if take.any() else None,
                               **{kind: int(np.sum(take & (label == kind))) for kind in DETECTION_TYPES}))
        image_errors = rows(output / ('round_%02d' % index) / 'acquisition_image_errors.csv')
        for group in ('zero_detections', 'has_detections'):
            subset = [row for row in image_errors if
                      (sum(int(row.get(k) or 0) for k in DETECTION_TYPES + ('Ignored',)) == 0)
                      == (group == 'zero_detections')]
            visibility.append(dict(round=index, group=group, images=len(subset),
                selected_images=sum(row['selected'] == 'True' for row in subset),
                gt=sum(int(row.get('GT') or 0) for row in subset),
                fn=sum(int(row.get('FN') or 0) for row in subset),
                miss=sum(int(row.get('Miss') or 0) for row in subset)))
    write_csv(output / 'candidate_reproduction.csv', checks)
    write_csv(output / 'predictor_class_types.csv', class_types)
    write_csv(output / 'predictor_calibration_bias.csv', calibration)
    write_csv(output / 'predictor_probability_strata.csv', strata)
    write_csv(output / 'acquisition_visibility.csv', visibility)
    print('All six saved candidate lists reproduce within 1e-9.', flush=True)


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    test = rows(output / 'test_rounds.csv')
    predictor = rows(output / 'predictor_rounds.csv')
    types = rows(output / 'predictor_types.csv')
    batches = rows(output / 'selected_batch_errors.csv')
    if len(test) != 7 or len(predictor) != 6:
        raise ValueError('Report requires all 7 test rounds and 6 acquisition rounds')
    transitions = []
    for first, second in zip(test[:-1], test[1:]):
        r = int(first['round'])
        for kind in ERRORS:
            batch = next(x for x in batches if int(x['round']) == r and x['type'] == kind)
            transitions.append(dict(acquisition_round=r, evaluated_next_round=r + 1, type=kind,
                acquired_count=int(batch['selected_count']),
                acquired_enrichment=float(batch['count_enrichment']),
                acquired_gt_normalized_enrichment=float(batch['gt_normalized_enrichment']),
                test_dap_before=float(first['dap_' + kind]),
                test_dap_after=float(second['dap_' + kind]),
                test_dap_change=float(second['dap_' + kind]) - float(first['dap_' + kind]),
                native_ap50_gain=float(second['native_ap50']) - float(first['native_ap50'])))
    write_csv(output / 'batch_to_test_transitions.csv', transitions)
    r = [int(x['round']) for x in test]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(r, [float(x['native_ap50']) for x in test], 'o-', label='VOC07 AP50')
    axes[0].plot(r, [float(x['native_ap75']) for x in test], 'o-', label='VOC07 AP75')
    axes[0].set(xlabel='Round', ylabel='AP (%)', title='PAL: test detection accuracy')
    axes[0].legend()
    for kind in ERRORS:
        axes[1].plot(r, [float(x['dap_' + kind]) for x in test], 'o-', label=kind)
    axes[1].set(xlabel='Round', ylabel='TIDE dAP50 (percentage points)',
                title='Residual errors (independent oracles)')
    axes[1].legend(ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / 'test_error_trajectory.png', dpi=180)
    plt.close(fig)
    last = [x for x in types if int(x['round']) == 6]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    kinds = [x['type'] for x in last if x['type'] != 'TP']
    axes[0].bar(kinds, [float(x['auc_vs_tp']) for x in last if x['type'] != 'TP'])
    axes[0].axhline(.5, color='gray', linestyle='--')
    axes[0].set(ylim=(0, 1), ylabel='AUROC: error vs TP', title='Round 6: PAL predictor discrimination')
    axes[1].bar([x['type'] for x in last], [100 * float(x['top10_coverage']) for x in last])
    axes[1].axhline(10, color='gray', linestyle='--', label='Uniform coverage: 10%')
    axes[1].set(ylabel='Coverage of subtype (%)', title='Round 6: top-10% LIUS entropy')
    axes[1].legend()
    fig.savefig(output / 'predictor_round6.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for kind in ERRORS:
        subset = [x for x in batches if x['type'] == kind]
        ax.plot([int(x['round']) for x in subset],
                [float(x['gt_normalized_enrichment']) for x in subset], 'o-', label=kind)
    ax.axhline(1, color='gray', linestyle='--')
    ax.set(xlabel='Acquisition round', ylabel='Selected / pool error count per GT',
           title='Actual PAL batches: GT-normalized enrichment')
    ax.legend(ncol=3)
    ax.grid(alpha=.2)
    fig.savefig(output / 'selected_batch_enrichment.png', dpi=180)
    plt.close(fig)
    calibration = rows(output / 'predictor_calibration.csv')
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    for index in (1, 6):
        bins = [x for x in calibration if int(x['round']) == index]
        ax.plot([float(x['predicted']) for x in bins],
                [float(x['observed']) for x in bins], 'o-', label='Round %d' % index)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted FP probability',
           ylabel='Observed FP fraction', title='PAL probability calibration')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / 'predictor_calibration.png', dpi=180)
    plt.close(fig)
    write_json(output / 'completion.json', dict(test_rounds=7, acquisition_rounds=6,
        max_native_ap50_reproduction_difference_pp=max(abs(float(x['reproduction_difference_pp'])) for x in test),
        max_candidate_score_difference=max(float(x['max_score_difference']) for x in rows(output / 'candidate_reproduction.csv'))))
    print('Transition tables and four summary figures completed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('predictor', 'report'), required=True)
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Analysis output must be outside the source run')
    if args.stage == 'predictor':
        predictor_details(source, output)
    else:
        report(output)


if __name__ == '__main__':
    main()
