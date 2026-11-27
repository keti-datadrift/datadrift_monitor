import numpy as np


def gaussian_scores(
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
    _, logdet = np.linalg.slogdet(cov)
    dim = train_features.shape[1]
    constant = dim * np.log(2.0 * np.pi) + logdet

    scores = []
    for x in test_features:
        diff = x[None, :] - means
        mahal = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
        log_likelihood = -0.5 * (constant + mahal)
        scores.append(float(-np.max(log_likelihood)))
    return np.asarray(scores, dtype=np.float64)
