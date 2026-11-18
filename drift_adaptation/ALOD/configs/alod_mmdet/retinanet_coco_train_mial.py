_base_ = './retinanet_coco_base.py'

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadMIAL',
        mial_lambda=0.5,
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(ann_file=None),
)

evaluation = dict(interval=999999999, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=26)
checkpoint_config = dict(interval=26, max_keep_ckpts=1, by_epoch=True)

mial_topk = 10000
mial_phase_schedule = dict(
    outer_loops=2,
    epoch_ratio=[3, 1],
    repeat_factor=2,
)
