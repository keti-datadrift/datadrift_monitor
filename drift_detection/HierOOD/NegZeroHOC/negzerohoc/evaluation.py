import torch


def get_id_classes(id_classes_fn):
    with open(id_classes_fn, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return sorted(lines)


def build_hierarchy(repo_root, id_split, hierarchy_path):
    from negzerohoc.prohoc_compat.hierarchy import Hierarchy

    id_classes = get_id_classes(str(id_split))
    return Hierarchy(id_classes, str(hierarchy_path)), id_classes


def node_labels_from_feature_targets(hierarchy, classes: list[str], targets: torch.Tensor) -> torch.Tensor:
    ds2node = hierarchy.gen_ds2node_map(classes)
    return ds2node[targets.long()]


def get_results(preds, node_labels, hierarchy, dists_mats=None):
    from negzerohoc.prohoc_compat import hierarchy_metrics as hm
    from negzerohoc.prohoc_compat.utils.hierarchy_utils import get_avg_hdist

    if dists_mats is not None:
        dists_mats = tuple(x.long() for x in dists_mats)

    hmet = hm.HierarchicalPredAccuracy(hierarchy, track_hdist=True)
    hmet.update_state(preds.long(), node_labels.long(), dists_mats=dists_mats)
    hd = hmet.result_hierarchy_distances()
    return {
        "acc": hmet.result(),
        "balanced_acc": hmet.result_balanced_accuracy(),
        "hdist": hd,
        "avg_hdist": get_avg_hdist(hd),
        "balanced_hdist": hmet.result_balanced_hierarchy_distance(),
        "class_hdists": hmet.result_class_hdists(),
    }


def make_distance_mats(hierarchy, device="cpu"):
    from negzerohoc.prohoc_compat.utils.hierarchy_utils import get_hdist_matrix

    gt_dists_mat, pred_dists_mat = get_hdist_matrix(
        hierarchy,
        range(len(hierarchy.id_node_list)),
        return_pair=True,
    )
    return gt_dists_mat.long().to(device), pred_dists_mat.long().to(device)


def evaluate_split(hierarchy, feature_payload, preds, dists_mats=None):
    node_labels = node_labels_from_feature_targets(
        hierarchy,
        feature_payload["classes"],
        feature_payload["targets"],
    )
    metrics = get_results(preds.cpu(), node_labels.cpu(), hierarchy, dists_mats=dists_mats)
    return node_labels, metrics


def mixed_summary(val_metrics: dict, ood_metrics: dict) -> dict:
    return {
        "mixed_balanced_acc": 0.5 * (float(val_metrics["balanced_acc"]) + float(ood_metrics["balanced_acc"])),
        "mixed_balanced_hdist": 0.5 * (
            float(val_metrics["balanced_hdist"]) + float(ood_metrics["balanced_hdist"])
        ),
    }
