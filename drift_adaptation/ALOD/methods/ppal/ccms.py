"""PPAL CCMS/diversity acquisition step."""

import numpy as np

from methods.common.feature_artifacts import filter_feature_artifact, load_feature_artifact
from methods.common.image_identity import (
    normalize_image_ids,
    validate_image_ids_subset,
)
from methods.ppal.base import BaseALSampler
from methods.ppal.distance import compute_ppal_image_distance_matrix


eps = 1e-10


class DiversitySampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        dataset_type,
        seed=None,
    ):
        super(DiversitySampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.kmeans_iterations = 100
        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else np.random

    @staticmethod
    def k_centroid_greedy(dis_matrix, K, rng=None):
        if rng is None:
            rng = np.random
        N = dis_matrix.shape[0]
        centroids = []
        c = rng.randint(0, N, (1,))[0]
        centroids.append(c)
        i = 1
        while i < K:
            centroids_diss = dis_matrix[:, centroids].copy()
            centroids_diss = centroids_diss.min(axis=1)
            centroids_diss[centroids] = -1
            new_c = np.argmax(centroids_diss)
            centroids.append(new_c)
            i += 1
        return centroids

    @staticmethod
    def kmeans(dis_matrix, K, n_iter=100, rng=None):
        N = dis_matrix.shape[0]
        centroids = DiversitySampler.k_centroid_greedy(dis_matrix, K, rng=rng)
        data_indices = np.arange(N)

        assign_dis_records = []
        for _ in range(n_iter):
            centroid_dis = dis_matrix[:, centroids]
            cluster_assign = np.argmin(centroid_dis, axis=1)
            assign_dis = centroid_dis.min(axis=1).sum()
            assign_dis_records.append(assign_dis)

            new_centroids = []
            for i in range(K):
                cluster_i = data_indices[cluster_assign == i]
                assert len(cluster_i) >= 1
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)
            centroids = np.array(new_centroids)
        return centroids.tolist()

    def al_acquisition(self, feature_artifact_path):
        feature_artifact = load_feature_artifact(
            feature_artifact_path,
            require_detection_features=True,
            require_image_features=False,
        )
        oracle_image_ids = normalize_image_ids(feature_artifact.image_ids)
        validate_image_ids_subset(
            oracle_image_ids,
            self.oracle_data.keys(),
            'PPAL feature artifact',
        )
        feature_artifact = filter_feature_artifact(
            feature_artifact,
            oracle_image_ids,
            artifact_name='PPAL feature artifact',
            require_all=True,
        )
        image_dis_matrix = compute_ppal_image_distance_matrix(feature_artifact)

        centroids = DiversitySampler.kmeans(
            image_dis_matrix,
            K=self.n_images,
            n_iter=self.kmeans_iterations,
            rng=self.random_state,
        )

        sampled_img_ids = [oracle_image_ids[index] for index in centroids]
        centroid_ranks = {
            index: rank for rank, index in enumerate(centroids, start=1)
        }
        candidate_records = []
        for index, image_id in enumerate(oracle_image_ids):
            centroid_rank = centroid_ranks.get(index)
            candidate_records.append({
                'image_id': image_id,
                'rank': index + 1,
                'score': None,
                'source': 'ccms',
                'components': {},
                'metadata': {
                    'distance_index': index,
                    'selected_by_ccms': centroid_rank is not None,
                    'centroid_rank': centroid_rank,
                },
            })

        metrics = {
            'feature_artifact': str(feature_artifact_path),
            'distance_matrix_shape': list(image_dis_matrix.shape),
            'distance_image_count': len(feature_artifact.image_ids),
            'canonical_distance_image_count': len(oracle_image_ids),
            'selected_candidate_count': len(sampled_img_ids),
            'kmeans_iterations': int(self.kmeans_iterations),
            'seed': self.seed,
        }
        return sampled_img_ids, candidate_records, metrics

    def al_round(self, feature_artifact_path, last_label_path):
        self.round += 1
        self.latest_labeled = last_label_path
        sampled_img_ids, candidate_records, metrics = self.al_acquisition(feature_artifact_path)
        return {
            'selected_image_ids': sampled_img_ids,
            'candidate_records': candidate_records,
            'selected_count': len(sampled_img_ids),
            'metrics': metrics,
        }
