import logging
import logging.config
import math
import platform

import numpy as np
import pkg_resources as pkg
import torch


LOGGING_NAME = "yolov5"


def emojis(text=""):
    return str(text).encode().decode("ascii", "ignore") if platform.system() == "Windows" else str(text)


def set_logging(name=LOGGING_NAME, verbose=True):
    level = logging.INFO if verbose else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {name: {"class": "logging.StreamHandler", "formatter": name, "level": level}},
            "loggers": {name: {"level": level, "handlers": [name], "propagate": False}},
        }
    )


set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, (lambda log_fn: lambda x: log_fn(emojis(x)))(fn))


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    if hard:
        assert result, f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"
    if verbose and not result:
        LOGGER.warning(f"WARNING: {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed")
    return result


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def colorstr(*input):
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
