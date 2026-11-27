"""PPAL CCMS image-distance computation from generic feature artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from methods.common.feature_artifacts import FeatureArtifact


INF = 1e12


def _torch_device(value: Any = None) -> str:
    if value:
        return str(value)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PPAL distance computation requires PyTorch') from exc
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_ppal_image_distance_matrix(
    artifact: FeatureArtifact,
    score_thr: float = 0.05,
    same_label: bool = True,
    metric: str = 'cosine',
    device: Any = None,
) -> np.ndarray:
    """Compute the PPAL CCMS image distance matrix.

    This mirrors the original PPAL detection-level distance: same-class
    detection features are matched by minimum cosine distance and weighted by
    detection confidence.
    """

    if metric != 'cosine':
        raise NotImplementedError('Only cosine PPAL distance is supported')
    if artifact.det_labels is None or artifact.det_scores is None or artifact.det_features is None:
        raise ValueError('PPAL distance requires detection labels, scores, and features')

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError('PPAL distance computation requires PyTorch') from exc

    selected_device = _torch_device(device)
    labels = torch.as_tensor(artifact.det_labels, dtype=torch.long, device=selected_device)
    scores = torch.as_tensor(artifact.det_scores, dtype=torch.float32, device=selected_device)
    feats = torch.as_tensor(artifact.det_features, dtype=torch.float32, device=selected_device)
    if artifact.det_valid is not None:
        valid = torch.as_tensor(artifact.det_valid, dtype=torch.bool, device=selected_device)
        scores = torch.where(valid, scores, torch.zeros_like(scores))

    n_images = labels.size(0)
    n_dets = labels.size(1)
    feat_dim = feats.size(-1)
    det_indices = torch.arange(n_dets, device=selected_device)
    feats = F.normalize(feats, p=2, dim=-1)
    feats_t = feats.transpose(1, 2)

    score_valid = (scores > float(score_thr)).to(dtype=feats.dtype)
    score_valid_t = score_valid[:, :, None].transpose(1, 2)
    labels_t = labels[:, :, None].transpose(1, 2)

    rows = []
    for i in range(n_images):
        labels_i = labels[i]
        scores_valid_i = score_valid[i]
        scores_i = scores[i]
        feats_i = feats[i]

        distances_i = -1.0 * torch.matmul(
            feats_i.view(1, n_dets, feat_dim),
            feats_t,
        ) + 1.0
        distances_i[:, det_indices, det_indices] = 0.0
        valid_scores = torch.matmul(
            scores_valid_i.view(1, n_dets, 1),
            score_valid_t,
        )

        if same_label:
            labels_i = labels_i[:, None].repeat(1, n_dets)
            valid_labels = (labels_i.view(1, n_dets, n_dets) == labels_t).to(dtype=feats.dtype)
        else:
            valid_labels = torch.ones_like(valid_scores)

        distances_i[(1.0 - valid_labels).to(dtype=torch.bool)] = 2.0
        distances_i[(1.0 - valid_scores).to(dtype=torch.bool)] = INF
        distances_i = distances_i.min(dim=-1)[0]

        norm = (valid_scores.max(dim=-1)[0] * scores_i[None, :]).sum(dim=-1) + 0.00001
        distances_i[distances_i > 2.0] = 0.0
        distances_i = distances_i * scores_i[None, :]
        distances_i = distances_i.sum(dim=-1) / norm
        rows.append(distances_i.detach().cpu())

    matrix = torch.stack(rows, dim=0)
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    return matrix.numpy().astype(np.float64, copy=False)
