_base_ = './retinanet_coco_base.py'

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaDetectionFeatureExportHead',
        total_images=0,
        max_det=100,
        feat_dim=256,
        output_path='',
    ),
)

data = dict(
    test=dict(
        type='ALCocoDataset',
        ann_file=None,
        img_prefix='data/coco/train2017/'),
)
