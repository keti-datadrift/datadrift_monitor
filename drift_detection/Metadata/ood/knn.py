import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_scores(train_features: np.ndarray, test_features: np.ndarray, k: int = 5) -> np.ndarray:
    k = min(k, len(train_features))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_features)
    distances, _ = nn.kneighbors(test_features)
    return distances.mean(axis=1)
