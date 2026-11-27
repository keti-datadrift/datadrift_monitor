from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def replay_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Seen-class CE used by PyCIL Replay on new data plus memory."""

    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits must be a matrix and targets a vector")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets have different batch sizes")
    return F.cross_entropy(logits, targets)


VALID_METHODS = {
    "replay_ce",
    "global_hap",
    "flat_lrhap",
    "sacil_v0",
    "logit_kd",
    "logit_kd_global_hap",
    "logit_kd_sacil_v0",
    "logit_kd_topkd",
    "afc",
    "afc_topkd",
    "afc_topkd_dual_rebalance",
    "afc_global_hap",
    "afc_flat_lrhap",
    "afc_sacil_v0",
    "takp_base",
    "takp",
}

AFC_METHODS = {
    "afc",
    "afc_topkd",
    "afc_topkd_dual_rebalance",
    "afc_global_hap",
    "afc_flat_lrhap",
    "afc_sacil_v0",
}

NON_GEOMETRY_METHODS = {
    "replay_ce",
    "logit_kd",
    "logit_kd_topkd",
    "takp_base",
    "takp",
    "afc",
    "afc_topkd",
    "afc_topkd_dual_rebalance",
}

LOGIT_KD_METHODS = {
    "logit_kd",
    "logit_kd_global_hap",
    "logit_kd_sacil_v0",
    "logit_kd_topkd",
    "takp",
}

TOPKD_METHODS = {
    "logit_kd_topkd",
    "afc_topkd",
    "afc_topkd_dual_rebalance",
    "takp",
}

DUAL_REBALANCING_METHODS = {
    "afc_topkd_dual_rebalance",
    "takp_base",
    "takp",
}

TAKP_METHODS = {"takp_base", "takp"}


def method_uses_geometry(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name not in NON_GEOMETRY_METHODS


def method_uses_afc(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name in AFC_METHODS


def method_uses_logit_kd(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name in LOGIT_KD_METHODS


def method_uses_topkd(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name in TOPKD_METHODS


def method_uses_dual_rebalancing(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name in DUAL_REBALANCING_METHODS


def method_uses_takp(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name in TAKP_METHODS
