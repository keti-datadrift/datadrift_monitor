from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .tree import HierarchyTree, TreeNode


@dataclass(frozen=True)
class _Cluster:
    node_id: str
    positions: tuple[int, ...]
    class_ids: tuple[int, ...]


class GriffinPeronaGreedy:
    """Bottom-up maximum-mutual-confusion taxonomy construction."""

    def build(
        self, class_ids: Sequence[int], affinity: Tensor
    ) -> HierarchyTree:
        order = tuple(int(value) for value in class_ids)
        count = len(order)
        if count == 0:
            raise ValueError("taxonomy requires at least one class")
        if affinity.shape != (count, count):
            raise ValueError("affinity shape does not match class count")
        affinity = affinity.detach().cpu().double()
        if not torch.isfinite(affinity).all():
            raise ValueError("affinity contains non-finite values")
        if not torch.allclose(affinity, affinity.t(), atol=1e-8):
            raise ValueError("affinity must be symmetric")

        nodes: dict[str, TreeNode] = {}
        clusters: list[_Cluster] = []
        for position, class_id in enumerate(order):
            node_id = HierarchyTree.leaf_node_id(class_id)
            nodes[node_id] = TreeNode(
                node_id=node_id,
                members=(class_id,),
                class_id=class_id,
            )
            clusters.append(
                _Cluster(node_id, (position,), (class_id,))
            )

        merge_index = 0
        while len(clusters) > 1:
            left_index, right_index = self._best_pair(clusters, affinity)
            left = clusters[left_index]
            right = clusters[right_index]
            if right.class_ids < left.class_ids:
                left, right = right, left
            node_id = f"node:{merge_index:04d}"
            members = tuple(sorted(left.class_ids + right.class_ids))
            positions = tuple(sorted(left.positions + right.positions))
            nodes[node_id] = TreeNode(
                node_id=node_id,
                members=members,
                left=left.node_id,
                right=right.node_id,
            )
            for index in sorted(
                (left_index, right_index), reverse=True
            ):
                del clusters[index]
            clusters.append(_Cluster(node_id, positions, members))
            clusters.sort(key=lambda cluster: cluster.class_ids)
            merge_index += 1

        return HierarchyTree(nodes, clusters[0].node_id, order)

    @staticmethod
    def _best_pair(
        clusters: Sequence[_Cluster], affinity: Tensor
    ) -> tuple[int, int]:
        best_score = -float("inf")
        best_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        best_pair: tuple[int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                first = clusters[left]
                second = clusters[right]
                rows = torch.tensor(first.positions, dtype=torch.long)
                columns = torch.tensor(second.positions, dtype=torch.long)
                score = float(
                    affinity.index_select(0, rows)
                    .index_select(1, columns)
                    .mean()
                    .item()
                )
                key = tuple(sorted((first.class_ids, second.class_ids)))
                if (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and (best_key is None or key < best_key)
                    )
                ):
                    best_score = score
                    best_key = key
                    best_pair = (left, right)
        if best_pair is None:
            raise RuntimeError("could not select a cluster pair")
        return best_pair

