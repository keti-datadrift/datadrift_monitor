from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TreeNode:
    node_id: str
    members: tuple[int, ...]
    left: str | None = None
    right: str | None = None
    class_id: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.class_id is not None


class HierarchyTree:
    def __init__(
        self,
        nodes: Mapping[str, TreeNode],
        root_id: str,
        class_order: Iterable[int],
    ) -> None:
        self.nodes = dict(nodes)
        self.root_id = str(root_id)
        self.class_order = tuple(int(value) for value in class_order)
        self._parent = self._build_parent_map()
        self._validate()

    @staticmethod
    def leaf_node_id(class_id: int) -> str:
        return f"leaf:{int(class_id)}"

    def _build_parent_map(self) -> dict[str, str]:
        parent: dict[str, str] = {}
        for node in self.nodes.values():
            for child in (node.left, node.right):
                if child is not None:
                    if child in parent:
                        raise ValueError(f"node has multiple parents: {child}")
                    parent[child] = node.node_id
        return parent

    def _validate(self) -> None:
        if self.root_id not in self.nodes:
            raise ValueError("root is missing from node mapping")
        leaf_ids = {
            node.class_id
            for node in self.nodes.values()
            if node.is_leaf and node.class_id is not None
        }
        if leaf_ids != set(self.class_order):
            raise ValueError("tree leaves do not match class order")
        if self.root_id in self._parent:
            raise ValueError("root cannot have a parent")
        for node in self.nodes.values():
            if node.is_leaf:
                if node.left is not None or node.right is not None:
                    raise ValueError("leaf cannot have children")
                if node.members != (node.class_id,):
                    raise ValueError("leaf members are invalid")
            else:
                if node.left not in self.nodes or node.right not in self.nodes:
                    raise ValueError("internal node must have two valid children")
                expected = tuple(
                    sorted(
                        self.nodes[node.left].members
                        + self.nodes[node.right].members
                    )
                )
                if node.members != expected:
                    raise ValueError("internal-node members are inconsistent")
        if set(self.nodes) - {self.root_id} != set(self._parent):
            raise ValueError("tree is disconnected")

    @property
    def num_leaves(self) -> int:
        return len(self.class_order)

    def leaf_node_ids(self) -> tuple[str, ...]:
        return tuple(self.leaf_node_id(class_id) for class_id in self.class_order)

    def internal_node_ids(self, include_root: bool = True) -> tuple[str, ...]:
        node_ids = [
            node_id for node_id, node in self.nodes.items() if not node.is_leaf
        ]
        node_ids.sort(key=self._internal_sort_key)
        if not include_root:
            node_ids = [node_id for node_id in node_ids if node_id != self.root_id]
        return tuple(node_ids)

    @staticmethod
    def _internal_sort_key(node_id: str) -> tuple[int, str]:
        suffix = node_id.split(":", maxsplit=1)[-1]
        try:
            return int(suffix), node_id
        except ValueError:
            return 2**31 - 1, node_id

    def descendants(self, node_id: str) -> tuple[int, ...]:
        return self.nodes[node_id].members

    def parent(self, node_id: str) -> str | None:
        return self._parent.get(node_id)

    def distance_from_ancestor(
        self, ancestor_id: str, descendant_id: str
    ) -> int:
        if ancestor_id == descendant_id:
            return 0
        distance = 0
        current = descendant_id
        while current in self._parent:
            current = self._parent[current]
            distance += 1
            if current == ancestor_id:
                return distance
        raise ValueError(
            f"{ancestor_id} is not an ancestor of {descendant_id}"
        )

    def state_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "class_order": list(self.class_order),
            "nodes": [
                {
                    **asdict(self.nodes[node_id]),
                    "members": list(self.nodes[node_id].members),
                }
                for node_id in sorted(self.nodes)
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "HierarchyTree":
        nodes = {
            item["node_id"]: TreeNode(
                node_id=item["node_id"],
                members=tuple(int(value) for value in item["members"]),
                left=item.get("left"),
                right=item.get("right"),
                class_id=(
                    None
                    if item.get("class_id") is None
                    else int(item["class_id"])
                ),
            )
            for item in state["nodes"]
        }
        return cls(nodes, state["root_id"], state["class_order"])

