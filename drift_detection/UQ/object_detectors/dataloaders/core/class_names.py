coco_names = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

pascal_voc_names = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

kitti_names = [
    "car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc",
]

bdd100k_names = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign",
]

cityscapes_names = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

foggy_cityscapes_names = list(cityscapes_names)

DATASET_CLASS_NAMES = {
    "coco": [name for name in coco_names[1:] if name != "N/A"],
    "voc": pascal_voc_names[1:],
    "pascal_voc": pascal_voc_names[1:],
    "kitti": kitti_names,
    "bdd100k": bdd100k_names,
    "bdd": bdd100k_names,
    "cityscapes": cityscapes_names,
    "foggy_cityscapes": foggy_cityscapes_names,
    "foggy_city": foggy_cityscapes_names,
}
