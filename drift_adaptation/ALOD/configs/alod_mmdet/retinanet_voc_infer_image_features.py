_base_ = './retinanet_voc_base.py'

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaImageFeatureExportHead',
        total_images=0,
        feat_dim=256,
        output_path='',
    ),
)

data = dict(
    test=dict(
        type='ALCocoDataset',
        ann_file=None,
        img_prefix='data/VOC0712/images/',
        classes=CLASSES),
)
