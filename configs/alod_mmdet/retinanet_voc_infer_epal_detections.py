_base_ = './retinanet_voc_base.py'

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadEPAL',
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=200,
        support_iou_thr=0.5,
        support_box_equal_tol=1e-6),
)

data = dict(
    test=dict(
        type='EPALFeatureCocoDataset',
        ann_file=None,
        img_prefix='data/VOC0712/images/',
        classes=CLASSES),
)
