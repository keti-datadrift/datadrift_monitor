import numpy as np


def mahalanobis_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    reg: float = 1e-5,
) -> np.ndarray:
    classes = np.unique(train_labels)
    means = np.stack([train_features[train_labels == c].mean(axis=0) for c in classes])
    centered = train_features - np.stack(
        [means[np.where(classes == y)[0][0]] for y in train_labels]
    )
    cov = np.cov(centered, rowvar=False)
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
    inv_cov = np.linalg.pinv(cov)

    scores = []
    for x in test_features:
        diff = x[None, :] - means
        dist = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
        scores.append(float(np.min(dist)))
    return np.asarray(scores, dtype=np.float64)
