from __future__ import annotations

import math
import hashlib
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


def streaming_file_identity(
    path: str | Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict:
    """Return a content-addressed identity without loading the file at once."""
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as source:
        while True:
            chunk = source.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    stat_size = int(resolved.stat().st_size)
    if size != stat_size:
        raise RuntimeError("File changed while its SHA-256 was computed")
    return {
        "canonical_path": str(resolved),
        "file_size": stat_size,
        "sha256": digest.hexdigest(),
    }


def canonical_named_tensor_hash(tensors: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact contiguous bytes."""
    if not tensors:
        raise ValueError("At least one tensor is required")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State value {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        metadata = (
            f"{name}\0{value.dtype}\0"
            + ",".join(str(int(size)) for size in value.shape)
            + "\0"
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = value.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def fold_weight_identity(checkpoint: dict) -> dict:
    lora = checkpoint.get("vision_lora_state_dict")
    proxies = checkpoint.get("metric_proxies")
    if not isinstance(lora, dict) or not lora:
        raise ValueError("Fold checkpoint has no LoRA tensor state")
    if not isinstance(proxies, torch.Tensor):
        raise ValueError("Fold checkpoint has no metric proxy tensor")
    lora_hash = canonical_named_tensor_hash(lora)
    proxy_hash = canonical_named_tensor_hash({
        "metric_proxies": proxies
    })
    return {
        "vision_lora_state_sha256": lora_hash,
        "metric_proxies_sha256": proxy_hash,
        "combined_weight_sha256": hashlib.sha256(
            f"{lora_hash}\0{proxy_hash}".encode("ascii")
        ).hexdigest(),
    }


def macro_terminal_weights(
    kinds: list[str],
    target_groups: list[str],
) -> torch.Tensor:
    """Give known and pseudo episodes equal macro-balanced total mass."""
    if len(kinds) != len(target_groups) or not kinds:
        raise ValueError("Episode kinds and target groups must align")
    if set(kinds) != {"known", "pseudo"}:
        raise ValueError("Both known and pseudo episodes are required")
    weights = torch.zeros(len(kinds), dtype=torch.float64)
    for kind in ("known", "pseudo"):
        group_counts = Counter(
            group
            for episode_kind, group in zip(kinds, target_groups)
            if episode_kind == kind
        )
        group_mass = 0.5 / float(len(group_counts))
        for index, (episode_kind, group) in enumerate(
            zip(kinds, target_groups)
        ):
            if episode_kind == kind:
                weights[index] = (
                    group_mass / float(group_counts[group])
                )
    if not torch.allclose(
        weights.sum(), torch.tensor(1.0, dtype=torch.float64)
    ):
        raise RuntimeError("Macro terminal weights are not normalized")
    return weights


def parent_descendant_mass(
    leaf_probabilities: torch.Tensor,
    hierarchy,
    leaf_nodes: list[str],
    parent_nodes: list[str],
) -> torch.Tensor:
    leaf_probabilities = leaf_probabilities.float()
    if int(leaf_probabilities.shape[1]) != len(leaf_nodes):
        raise ValueError("Leaf probabilities and leaf nodes differ")
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    columns = []
    for parent in parent_nodes:
        parent_index = node_to_index[parent]
        descendants = [
            leaf_index
            for leaf_index, leaf in enumerate(leaf_nodes)
            if parent_index in hierarchy.node_ancestors.get(leaf, [])
        ]
        if not descendants:
            raise ValueError(f"Parent {parent!r} has no retained leaf")
        columns.append(
            leaf_probabilities[:, descendants].sum(dim=1)
        )
    return torch.stack(columns, dim=1)


def route_preserving_terminal(
    leaf_probabilities: torch.Tensor,
    parent_mass: torch.Tensor,
    entcomp_unknown: torch.Tensor,
    *,
    leaf_node_indices: torch.Tensor,
    parent_node_indices: torch.Tensor,
    node_count: int,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
) -> torch.Tensor:
    """Construct CF-RPEP terminals while exactly preserving leaf odds."""
    leaf = leaf_probabilities
    mass = parent_mass.to(dtype=leaf.dtype, device=leaf.device)
    evidence = entcomp_unknown.to(
        dtype=leaf.dtype, device=leaf.device
    )
    if leaf.ndim != 2 or mass.ndim != 2 or evidence.shape != mass.shape:
        raise ValueError("CF-RPEP probabilities must be matrices")
    if int(leaf.shape[0]) != int(mass.shape[0]):
        raise ValueError("CF-RPEP sample counts differ")
    if int(leaf.shape[1]) != int(leaf_node_indices.numel()):
        raise ValueError("Leaf node indices differ from leaf probabilities")
    if int(mass.shape[1]) != int(parent_node_indices.numel()):
        raise ValueError("Parent node indices differ from parent evidence")
    if not torch.allclose(
        leaf.sum(dim=1),
        torch.ones(int(leaf.shape[0]), dtype=leaf.dtype, device=leaf.device),
        atol=1e-5,
    ):
        raise ValueError("Leaf classifier distribution is not normalized")
    a_tensor = torch.as_tensor(a, dtype=leaf.dtype, device=leaf.device)
    b_tensor = torch.as_tensor(b, dtype=leaf.dtype, device=leaf.device)
    if bool(a_tensor <= 0):
        raise ValueError("CF-RPEP scalar a must be positive")
    log_leaf = leaf.clamp_min(1e-30).log()
    log_parent_weight = (
        mass.clamp_min(1e-30).log()
        + a_tensor * torch.logit(
            evidence.clamp(1e-7, 1.0 - 1e-7)
        )
        + b_tensor
    )
    log_normalizer = torch.logsumexp(
        torch.cat([log_leaf, log_parent_weight], dim=1),
        dim=1,
        keepdim=True,
    )
    leaf_terminal = (log_leaf - log_normalizer).exp()
    parent_terminal = (log_parent_weight - log_normalizer).exp()
    terminal = torch.zeros(
        int(leaf.shape[0]),
        int(node_count),
        dtype=leaf.dtype,
        device=leaf.device,
    )
    terminal.index_copy_(
        1, leaf_node_indices.to(leaf.device), leaf_terminal
    )
    terminal.index_copy_(
        1, parent_node_indices.to(leaf.device), parent_terminal
    )
    sums = terminal.sum(dim=1)
    if not torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ):
        raise RuntimeError("CF-RPEP terminal distribution is not normalized")
    return terminal


def hierarchical_hazard_terminal(
    leaf_probabilities: torch.Tensor,
    entcomp_unknown: torch.Tensor,
    hierarchy,
    *,
    leaf_nodes: list[str],
    parent_nodes: list[str],
    leaf_node_indices: torch.Tensor,
    parent_node_indices: torch.Tensor,
    node_count: int,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
) -> torch.Tensor:
    """Convert local unknown evidence into a normalized stopping process."""
    leaf = leaf_probabilities
    evidence = entcomp_unknown.to(
        dtype=leaf.dtype, device=leaf.device
    )
    if leaf.ndim != 2 or evidence.ndim != 2:
        raise ValueError("HHP leaf probabilities/evidence must be matrices")
    if int(leaf.shape[0]) != int(evidence.shape[0]):
        raise ValueError("HHP sample counts differ")
    if int(leaf.shape[1]) != len(leaf_nodes):
        raise ValueError("HHP leaf node ordering is invalid")
    if int(evidence.shape[1]) != len(parent_nodes):
        raise ValueError("HHP parent evidence ordering is invalid")
    if int(leaf_node_indices.numel()) != len(leaf_nodes):
        raise ValueError("HHP leaf node indices are invalid")
    if int(parent_node_indices.numel()) != len(parent_nodes):
        raise ValueError("HHP parent node indices are invalid")
    if len(parent_nodes) != len(set(parent_nodes)):
        raise ValueError("HHP parent nodes must be unique")
    if not torch.allclose(
        leaf.sum(dim=1),
        torch.ones(
            int(leaf.shape[0]), dtype=leaf.dtype, device=leaf.device
        ),
        atol=1e-5,
    ):
        raise ValueError("HHP leaf classifier distribution is not normalized")

    a_tensor = torch.as_tensor(a, dtype=leaf.dtype, device=leaf.device)
    b_tensor = torch.as_tensor(b, dtype=leaf.dtype, device=leaf.device)
    if bool(a_tensor <= 0):
        raise ValueError("HHP scalar a must be positive")
    hazards = torch.sigmoid(
        a_tensor * torch.logit(evidence.clamp(1e-7, 1.0 - 1e-7))
        + b_tensor
    )
    return hierarchical_hazard_terminal_from_hazards(
        leaf,
        hazards,
        hierarchy,
        leaf_nodes=leaf_nodes,
        parent_nodes=parent_nodes,
        leaf_node_indices=leaf_node_indices,
        parent_node_indices=parent_node_indices,
        node_count=node_count,
    )


def hierarchical_hazard_terminal_from_hazards(
    leaf_probabilities: torch.Tensor,
    hazards: torch.Tensor,
    hierarchy,
    *,
    leaf_nodes: list[str],
    parent_nodes: list[str],
    leaf_node_indices: torch.Tensor,
    parent_node_indices: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """Build the HHP terminal distribution from direct local hazards."""
    leaf = leaf_probabilities
    hazards = hazards.to(dtype=leaf.dtype, device=leaf.device)
    if leaf.ndim != 2 or int(leaf.shape[1]) != len(leaf_nodes):
        raise ValueError("HHP direct leaf ordering is invalid")
    if int(leaf_node_indices.numel()) != len(leaf_nodes):
        raise ValueError("HHP direct leaf node indices are invalid")
    if int(parent_node_indices.numel()) != len(parent_nodes):
        raise ValueError("HHP direct parent node indices are invalid")
    if not torch.allclose(
        leaf.sum(dim=1),
        torch.ones(
            int(leaf.shape[0]), dtype=leaf.dtype, device=leaf.device
        ),
        atol=1e-5,
    ):
        raise ValueError("HHP direct leaf distribution is not normalized")
    if hazards.shape != (
        int(leaf.shape[0]),
        len(parent_nodes),
    ):
        raise ValueError("HHP direct hazard shape is invalid")
    if bool(((hazards < 0.0) | (hazards > 1.0)).any()):
        raise ValueError("HHP hazards must lie in [0,1]")
    parent_to_column = {
        parent: column for column, parent in enumerate(parent_nodes)
    }
    index_to_node = list(hierarchy.id_node_list)
    unknown_columns = [
        torch.zeros(
            int(leaf.shape[0]), dtype=leaf.dtype, device=leaf.device
        )
        for _ in parent_nodes
    ]
    leaf_columns = []
    for leaf_column, leaf_node in enumerate(leaf_nodes):
        remaining = leaf[:, leaf_column]
        path = [
            index_to_node[int(index)]
            for index in hierarchy.node_ancestors.get(leaf_node, [])
            if index_to_node[int(index)] in parent_to_column
        ]
        for parent in path:
            parent_column = parent_to_column[parent]
            stop = remaining * hazards[:, parent_column]
            unknown_columns[parent_column] = (
                unknown_columns[parent_column] + stop
            )
            remaining = remaining * (1.0 - hazards[:, parent_column])
        leaf_columns.append(remaining)
    leaf_terminal = torch.stack(leaf_columns, dim=1)
    parent_terminal = torch.stack(unknown_columns, dim=1)
    terminal = torch.zeros(
        int(leaf.shape[0]),
        int(node_count),
        dtype=leaf.dtype,
        device=leaf.device,
    )
    terminal.index_copy_(
        1, leaf_node_indices.to(leaf.device), leaf_terminal
    )
    terminal.index_copy_(
        1, parent_node_indices.to(leaf.device), parent_terminal
    )
    sums = terminal.sum(dim=1)
    if not torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ):
        raise RuntimeError("HHP terminal distribution is not normalized")
    return terminal


def leaf_coherent_entcomp_unknown(
    leaf_probabilities: torch.Tensor,
    hierarchy,
    *,
    leaf_nodes: list[str],
    parent_nodes: list[str],
) -> torch.Tensor:
    """Derive every parent EntComp value from one coherent leaf posterior."""
    leaf = leaf_probabilities.float()
    if leaf.ndim != 2 or int(leaf.shape[1]) != len(leaf_nodes):
        raise ValueError("Coherent EntComp leaf ordering is invalid")
    if not torch.allclose(
        leaf.sum(dim=1),
        torch.ones(
            int(leaf.shape[0]), dtype=leaf.dtype, device=leaf.device
        ),
        atol=1e-5,
    ):
        raise ValueError(
            "Coherent EntComp leaf distribution is not normalized"
        )
    if len(parent_nodes) != len(set(parent_nodes)):
        raise ValueError("Coherent EntComp parent nodes must be unique")
    allowed = set(parent_nodes) | set(leaf_nodes)
    index_to_node = list(hierarchy.id_node_list)
    leaf_paths = {
        leaf_node: [
            index_to_node[int(index)]
            for index in hierarchy.node_ancestors.get(leaf_node, [])
        ] + [leaf_node]
        for leaf_node in leaf_nodes
    }
    columns = []
    for parent in parent_nodes:
        branch_to_leaf_columns: dict[str, list[int]] = {}
        for leaf_column, leaf_node in enumerate(leaf_nodes):
            path = leaf_paths[leaf_node]
            if parent not in path:
                continue
            parent_position = path.index(parent)
            branch = next(
                (
                    node
                    for node in path[parent_position + 1:]
                    if node in allowed
                ),
                None,
            )
            if branch is None:
                raise RuntimeError(
                    f"Parent {parent!r} has no coherent child branch"
                )
            branch_to_leaf_columns.setdefault(branch, []).append(
                leaf_column
            )
        if len(branch_to_leaf_columns) < 2:
            raise ValueError(
                f"Parent {parent!r} has fewer than two visible branches"
            )
        branch_probabilities = torch.stack([
            leaf[:, indices].sum(dim=1)
            for _, indices in sorted(branch_to_leaf_columns.items())
        ], dim=1)
        group_sum = branch_probabilities.sum(dim=1)
        local = branch_probabilities / group_sum.clamp_min(1e-12)[:, None]
        entropy = -(
            local.clamp_min(1e-12) * local.clamp_min(1e-12).log()
        ).sum(dim=1)
        complement = (1.0 - group_sum).clamp(0.0, 1.0)
        score = entropy + complement
        columns.append(score / (group_sum + score).clamp_min(1e-12))
    return torch.stack(columns, dim=1)


RELATIONAL_HAZARD_FEATURE_NAMES = (
    "multidepth_entcomp_logit",
    "coherent_entcomp_logit",
    "descendant_mass_logit",
    "normalized_branch_entropy",
    "largest_local_branch_probability",
    "local_top1_top2_margin",
    "log_visible_child_count",
    "normalized_parent_depth",
    "global_leaf_max_probability",
    "normalized_global_leaf_entropy",
)


def relational_hazard_features(
    leaf_probabilities: torch.Tensor,
    multidepth_unknown: torch.Tensor,
    hierarchy,
    *,
    leaf_nodes: list[str],
    parent_nodes: list[str],
) -> torch.Tensor:
    """Build shared, node-agnostic relational features for every parent."""
    leaf = leaf_probabilities.float()
    multidepth = multidepth_unknown.float()
    if multidepth.shape != (
        int(leaf.shape[0]),
        len(parent_nodes),
    ):
        raise ValueError("Relational hazard evidence shape is invalid")
    coherent = leaf_coherent_entcomp_unknown(
        leaf,
        hierarchy,
        leaf_nodes=leaf_nodes,
        parent_nodes=parent_nodes,
    )
    allowed = set(parent_nodes) | set(leaf_nodes)
    index_to_node = list(hierarchy.id_node_list)
    leaf_paths = {
        leaf_node: [
            index_to_node[int(index)]
            for index in hierarchy.node_ancestors.get(leaf_node, [])
        ] + [leaf_node]
        for leaf_node in leaf_nodes
    }
    global_max = leaf.max(dim=1).values
    global_entropy = -(
        leaf.clamp_min(1e-12) * leaf.clamp_min(1e-12).log()
    ).sum(dim=1)
    global_entropy = global_entropy / math.log(max(2, len(leaf_nodes)))
    max_depth = max(
        1,
        max(
            len(hierarchy.node_ancestors.get(parent, []))
            for parent in parent_nodes
        ),
    )
    columns = []
    for parent_column, parent in enumerate(parent_nodes):
        branch_to_leaf_columns: dict[str, list[int]] = {}
        for leaf_column, leaf_node in enumerate(leaf_nodes):
            path = leaf_paths[leaf_node]
            if parent not in path:
                continue
            parent_position = path.index(parent)
            branch = next(
                (
                    node
                    for node in path[parent_position + 1:]
                    if node in allowed
                ),
                None,
            )
            if branch is None:
                raise RuntimeError(
                    f"Parent {parent!r} has no relational child branch"
                )
            branch_to_leaf_columns.setdefault(branch, []).append(
                leaf_column
            )
        child_count = len(branch_to_leaf_columns)
        if child_count < 2:
            raise ValueError(
                f"Parent {parent!r} has fewer than two relational branches"
            )
        branch = torch.stack([
            leaf[:, indices].sum(dim=1)
            for _, indices in sorted(branch_to_leaf_columns.items())
        ], dim=1)
        descendant_mass = branch.sum(dim=1)
        local = branch / descendant_mass.clamp_min(1e-12)[:, None]
        entropy = -(
            local.clamp_min(1e-12) * local.clamp_min(1e-12).log()
        ).sum(dim=1)
        normalized_entropy = entropy / math.log(child_count)
        top2 = local.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        depth = len(hierarchy.node_ancestors.get(parent, []))
        columns.append(torch.stack([
            torch.logit(
                multidepth[:, parent_column].clamp(1e-6, 1.0 - 1e-6)
            ),
            torch.logit(
                coherent[:, parent_column].clamp(1e-6, 1.0 - 1e-6)
            ),
            torch.logit(
                descendant_mass.clamp(1e-6, 1.0 - 1e-6)
            ),
            normalized_entropy,
            top2[:, 0],
            margin,
            torch.full_like(
                descendant_mass,
                math.log(child_count) / math.log(max(2, len(leaf_nodes))),
            ),
            torch.full_like(descendant_mass, depth / max_depth),
            global_max,
            global_entropy,
        ], dim=1))
    return torch.stack(columns, dim=1)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def fit_shared_route_scalars(
    bundles: list[dict],
    *,
    max_iter: int = 100,
) -> dict:
    """Fit global a,b by fixed full-batch strong-Wolfe L-BFGS."""
    if not bundles:
        raise ValueError("At least one calibration bundle is required")
    kinds = [
        kind for bundle in bundles for kind in bundle["kinds"]
    ]
    groups = [
        group
        for bundle in bundles
        for group in bundle["target_groups"]
    ]
    combined_weights = macro_terminal_weights(kinds, groups)
    offset = 0
    prepared = []
    for bundle in bundles:
        count = len(bundle["kinds"])
        weights = combined_weights[offset:offset + count]
        offset += count
        prepared.append({
            **bundle,
            "weights": weights,
            "leaf_probabilities": bundle[
                "leaf_probabilities"
            ].double(),
            "parent_mass": bundle["parent_mass"].double(),
            "entcomp_unknown": bundle["entcomp_unknown"].double(),
            "target_node_indices": bundle[
                "target_node_indices"
            ].long(),
        })

    raw_a = torch.nn.Parameter(torch.tensor(
        _inverse_softplus(1.0), dtype=torch.float64
    ))
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [raw_a, b],
        lr=1.0,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def objective(backward: bool) -> torch.Tensor:
        a = F.softplus(raw_a).clamp_min(1e-8)
        loss = torch.zeros((), dtype=torch.float64)
        for bundle in prepared:
            terminal = route_preserving_terminal(
                bundle["leaf_probabilities"],
                bundle["parent_mass"],
                bundle["entcomp_unknown"],
                leaf_node_indices=bundle["leaf_node_indices"],
                parent_node_indices=bundle["parent_node_indices"],
                node_count=int(bundle["node_count"]),
                a=a,
                b=b,
            )
            target_probability = terminal.gather(
                1, bundle["target_node_indices"].unsqueeze(1)
            ).squeeze(1)
            loss = loss - (
                bundle["weights"]
                * target_probability.clamp_min(1e-30).log()
            ).sum()
        if backward:
            loss.backward()
        return loss

    initial_loss = float(objective(False).detach())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        return objective(True)

    optimizer.step(closure)
    final_loss = float(objective(False).detach())
    state = optimizer.state[raw_a]
    return {
        "a": float(F.softplus(raw_a).detach()),
        "b": float(b.detach()),
        "initial_nll": initial_loss,
        "final_nll": final_loss,
        "max_iter": int(max_iter),
        "optimizer": "LBFGS",
        "line_search": "strong_wolfe",
        "iterations": int(state.get("n_iter", 0)),
        "function_evaluations": int(state.get("func_evals", 0)),
        "known_weight_sum": float(combined_weights[
            torch.tensor([kind == "known" for kind in kinds])
        ].sum()),
        "pseudo_weight_sum": float(combined_weights[
            torch.tensor([kind == "pseudo" for kind in kinds])
        ].sum()),
    }


def fit_shared_hazard_scalars(
    bundles: list[dict],
    hierarchy,
    *,
    max_iter: int = 100,
) -> dict:
    """Fit the two global HHP hazard-calibration scalars."""
    if not bundles:
        raise ValueError("At least one HHP calibration bundle is required")
    kinds = [kind for bundle in bundles for kind in bundle["kinds"]]
    groups = [
        group for bundle in bundles for group in bundle["target_groups"]
    ]
    combined_weights = macro_terminal_weights(kinds, groups)
    prepared = []
    offset = 0
    for bundle in bundles:
        count = len(bundle["kinds"])
        for key in ("retained_classes", "parent_nodes"):
            if key not in bundle:
                raise ValueError(f"HHP bundle is missing {key}")
        prepared.append({
            **bundle,
            "weights": combined_weights[offset:offset + count],
            "leaf_probabilities": bundle[
                "leaf_probabilities"
            ].double(),
            "entcomp_unknown": bundle["entcomp_unknown"].double(),
            "target_node_indices": bundle[
                "target_node_indices"
            ].long(),
        })
        offset += count

    raw_a = torch.nn.Parameter(torch.tensor(
        _inverse_softplus(1.0), dtype=torch.float64
    ))
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [raw_a, b],
        lr=1.0,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def objective(backward: bool) -> torch.Tensor:
        a = F.softplus(raw_a).clamp_min(1e-8)
        loss = torch.zeros((), dtype=torch.float64)
        for bundle in prepared:
            terminal = hierarchical_hazard_terminal(
                bundle["leaf_probabilities"],
                bundle["entcomp_unknown"],
                hierarchy,
                leaf_nodes=bundle["retained_classes"],
                parent_nodes=bundle["parent_nodes"],
                leaf_node_indices=bundle["leaf_node_indices"],
                parent_node_indices=bundle["parent_node_indices"],
                node_count=int(bundle["node_count"]),
                a=a,
                b=b,
            )
            probability = terminal.gather(
                1, bundle["target_node_indices"].unsqueeze(1)
            ).squeeze(1)
            loss = loss - (
                bundle["weights"]
                * probability.clamp_min(1e-30).log()
            ).sum()
        if backward:
            loss.backward()
        return loss

    initial_loss = float(objective(False).detach())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        return objective(True)

    optimizer.step(closure)
    final_loss = float(objective(False).detach())
    state = optimizer.state[raw_a]
    return {
        "a": float(F.softplus(raw_a).detach()),
        "b": float(b.detach()),
        "initial_nll": initial_loss,
        "final_nll": final_loss,
        "max_iter": int(max_iter),
        "optimizer": "LBFGS",
        "line_search": "strong_wolfe",
        "iterations": int(state.get("n_iter", 0)),
        "function_evaluations": int(state.get("func_evals", 0)),
        "known_weight_sum": float(combined_weights[
            torch.tensor([kind == "known" for kind in kinds])
        ].sum()),
        "pseudo_weight_sum": float(combined_weights[
            torch.tensor([kind == "pseudo" for kind in kinds])
        ].sum()),
    }
