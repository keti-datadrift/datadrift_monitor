import numpy as np
import torch
import torch.nn as nn
import time
import torchvision

from models.yolov5.core.models.experimental import attempt_load
from models.yolov5.core.utils.general import xywh2xyxy
from models.yolov5.core.utils.metrics import box_iou
from models.yolov5.core.utils.preprocess import letterbox


class YOLOV5TorchObjectDetector(nn.Module):
    def __init__(self,
                 model_weight,
                 device,
                 img_size,
                 names=None,
                 mode='eval',
                 confidence=0.25,
                 iou_thresh=0.45,
                 agnostic_nms=True,
                 fuse=True):
        super(YOLOV5TorchObjectDetector, self).__init__()
        self.device = device
        self.model = None
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.model = attempt_load(model_weight, device=device,fuse=fuse)
        print("[INFO] model is loaded")
        first_param = next(self.model.parameters(), None)
        requested_device = torch.device(device)
        if first_param is not None and first_param.device.type != requested_device.type:
            raise RuntimeError(
                f"YOLOv5 model loaded on {first_param.device}, expected device type {requested_device.type}."
            )
        if self.mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        if names is None:
            print('[INFO] fetching names from coco file')
            self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                          'cow',
                          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                          'frisbee',
                          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                          'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                          'apple',
                          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                          'couch',
                          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                          'keyboard', 'cell phone',
                          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                          'teddy bear',
                          'hair drier', 'toothbrush']
        else:
            self.names = names


        img = torch.zeros((1, 3, *self.img_size), device=device)
        with torch.no_grad():
            self.model(img)

    @staticmethod
    def non_max_suppression(prediction, logits, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300, return_indices=False):
        nc = prediction.shape[2] - 5
        xc = prediction[..., 4] > conf_thres


        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'


        min_wh, max_wh = 2, 4096
        max_nms = 30000
        time_limit = 10.0
        redundant = True
        multi_label &= nc > 1
        merge = False

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, nc), device=logits.device)] * logits.shape[0]
        objectivness_output = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
        index_output = [torch.zeros((0,), dtype=torch.long, device=prediction.device)] * prediction.shape[0]
        for xi, (x, log_) in enumerate(zip(prediction, logits)):


            candidate_mask = xc[xi]
            candidate_indices = torch.arange(prediction.shape[1], device=prediction.device)[candidate_mask]
            x = x[candidate_mask]
            log_ = log_[candidate_mask]

            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]
                v[:, 4] = 1.0
                v[range(len(l)), l[:, 0].long() + 5] = 1.0
                x = torch.cat((x, v), 0)


            if not x.shape[0]:
                continue


            objectivness = x.clone()
            objectivness = objectivness[:, 4:5].view(-1)
            x[:, 5:] = x[:, 5:] * x[:, 4:5]



            box = xywh2xyxy(x[:, :4])


            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim=True)

                conf_mask = conf.view(-1) > conf_thres
                x = torch.cat((box, conf, j.float()), 1)[conf_mask]
                log_ = log_[conf_mask]
                candidate_indices = candidate_indices[conf_mask]

            if classes is not None:
                class_mask = (x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)
                x = x[class_mask]
                log_ = log_[class_mask]
                candidate_indices = candidate_indices[class_mask]


            n = x.shape[0]
            if not n:
                continue
            elif n > max_nms:
                sorted_idx = x[:, 4].argsort(descending=True)[:max_nms]
                x = x[sorted_idx]
                log_ = log_[sorted_idx]
                objectivness = objectivness[sorted_idx]
                candidate_indices = candidate_indices[sorted_idx]


            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            if merge and (1 < n < 3E3):

                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]

            output[xi] = x[i]
            logits_output[xi] = log_[i]
            objectivness_output[xi] = objectivness[i]
            index_output[xi] = candidate_indices[i]
            assert log_[i].shape[0] == x[i].shape[0]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break

        if return_indices:
            return output, logits_output, objectivness_output, index_output
        return output, logits_output,objectivness_output

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):

        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def forward(self, img, return_features=False):
        model_output = self.model(img, augment=False)
        prediction = model_output[0]
        logits = model_output[1] if len(model_output) > 1 else None
        x = model_output[2] if len(model_output) > 2 else None
        prediction, logits,objectivness = self.non_max_suppression(prediction, logits, self.confidence, self.iou_thresh,
                                                      classes=None,
                                                      agnostic=self.agnostic)
        self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(img.shape[0])] for _ in
                                                                        range(4)]
        for i, det in enumerate(prediction):
            if len(det):
                for *xyxy, conf, cls in det:
                    bbox = self.box2box(xyxy)




                    self.boxes[i].append(bbox)
                    self.confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    self.classes[i].append(cls)
                    if self.names is not None:
                        self.class_names[i].append(self.names[cls])
                    else:
                        self.class_names[i].append(cls)
        features = x if return_features else None
        return [self.boxes, self.classes, self.class_names, self.confidences], logits, objectivness, features

    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)
        img = np.array([self.yolo_resize(im, new_shape=self.img_size)[0] for im in im0])
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0
        return img

    def box2box(self,bbox):
        x1 = int(bbox[0].detach().cpu().numpy())
        y1 = int(bbox[1].detach().cpu().numpy())
        x2 = int(bbox[2].detach().cpu().numpy())
        y2 = int(bbox[3].detach().cpu().numpy())
        return [x1,y1,x2,y2]
