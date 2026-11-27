_base_ = './retinanet_voc_base.py'

model = dict(
    bbox_head=dict(
        type='RetinaQualityEMAHead',
        base_momentum=0.8514577710948755,
    )
)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(ann_file=None),
)

evaluation = dict(interval=999999999, metric='bbox')
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
fp16 = dict(loss_scale='dynamic')

lr_config = dict(policy='step', warmup='linear', warmup_iters=32, warmup_ratio=0.001, step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=26)
checkpoint_config = dict(interval=26, max_keep_ckpts=1, by_epoch=True)
