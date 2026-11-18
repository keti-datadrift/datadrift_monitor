"""Shared vector math helpers."""

from __future__ import annotations

import math

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 vector, preserving zero vectors."""

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not math.isfinite(norm):
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 matrix, preserving zero rows."""

    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe = np.isfinite(norms) & (norms > 0.0)
    output = values.copy()
    output[safe[:, 0]] = output[safe[:, 0]] / norms[safe[:, 0]]
    return output


def squared_euclidean_to_centers(
    candidates: np.ndarray,
    centers: np.ndarray,
    batch_size: int = 512,
    center_batch_size: int = 2048,
) -> np.ndarray:
    """Return each candidate row's minimum squared Euclidean distance to centers."""

    candidate_values = np.asarray(candidates, dtype=np.float32)
    center_values = np.asarray(centers, dtype=np.float32)
    if candidate_values.ndim != 2 or center_values.ndim != 2:
        raise ValueError('candidates and centers must be 2D arrays')
    if candidate_values.shape[1] != center_values.shape[1]:
        raise ValueError(
            'candidate dim %d does not match center dim %d'
            % (candidate_values.shape[1], center_values.shape[1]))
    if center_values.shape[0] == 0:
        return np.full((candidate_values.shape[0], ), np.inf, dtype=np.float64)

    candidate_norms = np.sum(candidate_values.astype(np.float64) ** 2, axis=1)
    center_norms = np.sum(center_values.astype(np.float64) ** 2, axis=1)
    min_distances = np.full((candidate_values.shape[0], ), np.inf, dtype=np.float64)
    for start in range(0, candidate_values.shape[0], batch_size):
        end = min(start + batch_size, candidate_values.shape[0])
        batch = candidate_values[start:end].astype(np.float64, copy=False)
        batch_min = np.full((end - start, ), np.inf, dtype=np.float64)
        for center_start in range(0, center_values.shape[0], center_batch_size):
            center_end = min(center_start + center_batch_size, center_values.shape[0])
            center_batch = center_values[center_start:center_end].astype(np.float64, copy=False)
            distances = (
                candidate_norms[start:end, None]
                + center_norms[center_start:center_end][None, :]
                - 2.0 * np.matmul(batch, center_batch.T)
            )
            batch_min = np.minimum(batch_min, distances.min(axis=1))
        min_distances[start:end] = np.maximum(batch_min, 0.0)
    return min_distances


def squared_euclidean_to_vector(candidates: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances from each candidate row to one center."""

    candidate_values = np.asarray(candidates, dtype=np.float32)
    center_value = np.asarray(center, dtype=np.float32).reshape(1, -1)
    return np.maximum(
        np.sum((candidate_values.astype(np.float64) - center_value.astype(np.float64)) ** 2, axis=1),
        0.0,
    )
