
"""
Experimental modules
"""
import math
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from models.yolov5.core.models.common import Conv

from models.yolov5.core.utils.downloads import attempt_download


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super().__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        n = len(k)
        if equal_ch:
            i = torch.linspace(0, n - 1E-6, c2).floor()
            c_ = [(i == g).sum() for g in range(n)]
        else:
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]


        y = torch.cat(y, 1)
        return y, None


def _register_legacy_yolov5_pickle_modules():
    from models.yolov5.core.models import common as yolov5_common
    from models.yolov5.core.models import experimental as yolov5_experimental
    from models.yolov5.core.models import yolo as yolov5_yolo

    sys.modules.setdefault("models.common", yolov5_common)
    sys.modules.setdefault("models.experimental", yolov5_experimental)

    legacy_yolo_pkg = sys.modules.get("models.yolo")
    if legacy_yolo_pkg is None or not hasattr(legacy_yolo_pkg, "__path__"):
        legacy_yolo_pkg = types.ModuleType("models.yolo")
        legacy_yolo_pkg.__path__ = []
        sys.modules["models.yolo"] = legacy_yolo_pkg
    for attr_name in dir(yolov5_yolo):
        if not attr_name.startswith("__"):
            setattr(legacy_yolo_pkg, attr_name, getattr(yolov5_yolo, attr_name))
    legacy_yolo_pkg.common = yolov5_common
    legacy_yolo_pkg.experimental = yolov5_experimental
    legacy_yolo_pkg.yolo = yolov5_yolo

    legacy_models_pkg = sys.modules.get("models.yolo.models")
    if legacy_models_pkg is None or not hasattr(legacy_models_pkg, "__path__"):
        legacy_models_pkg = types.ModuleType("models.yolo.models")
        legacy_models_pkg.__path__ = []
        sys.modules["models.yolo.models"] = legacy_models_pkg
    legacy_yolo_pkg.models = legacy_models_pkg
    legacy_models_pkg.common = yolov5_common
    legacy_models_pkg.experimental = yolov5_experimental
    legacy_models_pkg.yolo = yolov5_yolo
    sys.modules.setdefault("models.yolo.common", yolov5_common)
    sys.modules.setdefault("models.yolo.experimental", yolov5_experimental)
    sys.modules.setdefault("models.yolo.yolo", yolov5_yolo)
    sys.modules.setdefault("models.yolo.models.common", yolov5_common)
    sys.modules.setdefault("models.yolo.models.experimental", yolov5_experimental)
    sys.modules.setdefault("models.yolo.models.yolo", yolov5_yolo)


def attempt_load(weights, device=None, inplace=True, fuse=True):



    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:


        ckpt_path = attempt_download(w)
        _register_legacy_yolov5_pickle_modules()
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:

            ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = (ckpt.get('ema') or ckpt['model']).float()
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())


    for m in model.modules():
        t = type(m)
        t_name = t.__name__
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU) or t_name in ("Detect", "Model"):
            m.inplace = inplace
            if t_name == "Detect" and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model

