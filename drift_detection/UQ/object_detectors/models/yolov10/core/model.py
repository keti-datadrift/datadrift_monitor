from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .modules import C2f, C2fCIB, Concat, Conv, PSA, SCDown, SPPF, initialize_yolov10_weights, v10Detect
from .ops import make_divisible


MODULES = {
    "Conv": Conv,
    "C2f": C2f,
    "SCDown": SCDown,
    "SPPF": SPPF,
    "PSA": PSA,
    "C2fCIB": C2fCIB,
    "Concat": Concat,
    "v10Detect": v10Detect,
    "nn.Upsample": nn.Upsample,
}


def _resolve_model_arg(d, arg):
    if isinstance(arg, str):
        lowered = arg.strip().lower()
        if lowered in {"none", "null"}:
            return None
        if arg in d:
            return d[arg]
    return arg


def parse_model(d, ch):
    depth, width, max_channels = 1.0, 1.0, float("inf")
    scales = d.get("scales")
    if scales:
        scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m_cls = MODULES[m] if isinstance(m, str) else m
        args = [_resolve_model_arg(d, arg) for arg in list(args)]
        n = max(round(n * depth), 1) if n > 1 else n
        if m_cls in {Conv, C2f, SCDown, SPPF, PSA, C2fCIB}:
            c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m_cls in {C2f, C2fCIB}:
                args.insert(2, n)
                n = 1
        elif m_cls is Concat:
            c2 = sum(ch[x] for x in f)
        elif m_cls is v10Detect:
            args.append([ch[x] for x in f])
            c2 = None
        elif m_cls is nn.Upsample:
            c2 = ch[f]
        else:
            raise ValueError(f"Unsupported YOLOv10 module: {m}")
        module = nn.Sequential(*(m_cls(*args) for _ in range(n))) if n > 1 else m_cls(*args)
        module.i, module.f, module.type = i, f, m_cls.__name__
        layers.append(module)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        if i == 0:
            ch = []
        ch.append(c2 if c2 is not None else ch[f[0]])
    return nn.Sequential(*layers), sorted(save)


class YOLOv10DetectionModel(nn.Module):
    def __init__(self, cfg, nc=80):
        super().__init__()
        if isinstance(cfg, (str, Path)):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        self.yaml = deepcopy(cfg)
        self.yaml["nc"] = int(nc)
        self.model, self.save = parse_model(self.yaml, ch=[3])
        self.names = [str(i) for i in range(int(nc))]
        self.nc = int(nc)
        self.inplace = self.yaml.get("inplace", True)
        self.args = type("Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5})()
        self._initialize_strides()
        initialize_yolov10_weights(self)

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def forward(self, x, augment=False, *args, **kwargs):
        return self._forward_once(x)

    def forward_features(self, x):
        y = []
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        head = self.model[-1]
        return y[head.f] if isinstance(head.f, int) else [x if j == -1 else y[j] for j in head.f]

    def forward_one2one(self, x):
        return self.forward_one2one_from_features(self.forward_features(x))

    def forward_one2one_from_features(self, features):
        return {"one2many": None, "one2one": self.model[-1].forward_one2one(features)}

    def _initialize_strides(self):
        m = self.model[-1]
        if isinstance(m, v10Detect):
            s = 256
            m.inplace = self.inplace
            was_training = self.training
            self.train()
            with torch.no_grad():
                raw = self._forward_once(torch.zeros(1, 3, s, s))["one2many"]
            m.stride = torch.tensor([s / x.shape[-2] for x in raw])
            self.stride = m.stride
            m.bias_init()
            if not was_training:
                self.eval()
            else:
                self.train()


def load_yolov10_cfg(variant="n"):
    variant = str(variant or "n").lower().replace("yolov10", "")
    cfg_path = Path(__file__).resolve().parent / "cfg" / f"yolov10{variant}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"YOLOv10 cfg not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["YOLOv10DetectionModel", "load_yolov10_cfg", "parse_model"]
