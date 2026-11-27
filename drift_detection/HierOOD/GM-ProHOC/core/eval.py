from __future__ import annotations

from typing import Any

import torch

from libs import hierarchy_metrics as hm
from libs.utils.hierarchy_utils import get_avg_hdist, get_hdist_matrix


def evaluate_predictions(preds: torch.Tensor, targets: torch.Tensor, hierarchy) -> dict[str, Any]:
    gt_dists_mat, pred_dists_mat = get_hdist_matrix(
        hierarchy,
        range(len(hierarchy.id_node_list)),
        return_pair=True,
    )
    hmet = hm.HierarchicalPredAccuracy(hierarchy, track_hdist=True)
    hmet.update_state(
        preds.long().cpu(),
        targets.long().cpu(),
        dists_mats=(gt_dists_mat.long(), pred_dists_mat.long()),
    )
    hd = hmet.result_hierarchy_distances()
    return {
        "acc": hmet.result(),
        "balanced_acc": hmet.result_balanced_accuracy(),
        "hdist": hd,
        "avg_hdist": get_avg_hdist(hd),
        "balanced_hdist": hmet.result_balanced_hierarchy_distance(),
        "class_hdists": hmet.result_class_hdists(),
    }
