from __future__ import annotations

import torch
import torch.nn.functional as F

from negzerohoc.cf_rpep import (
    RELATIONAL_HAZARD_FEATURE_NAMES,
    hierarchical_hazard_terminal_from_hazards,
    macro_terminal_weights,
    relational_hazard_features,
)


def bundle_relational_features(bundle: dict, hierarchy) -> torch.Tensor:
    used_hierarchy = bundle.get("_hierarchy", hierarchy)
    return relational_hazard_features(
        bundle["leaf_probabilities"],
        bundle["entcomp_unknown"],
        used_hierarchy,
        leaf_nodes=bundle["retained_classes"],
        parent_nodes=bundle["parent_nodes"],
    )


def relational_hazard_probabilities(
    bundle: dict,
    hierarchy,
    model: dict,
) -> torch.Tensor:
    features = bundle_relational_features(bundle, hierarchy).double()
    mean = torch.as_tensor(model["feature_mean"], dtype=torch.float64)
    scale = torch.as_tensor(model["feature_scale"], dtype=torch.float64)
    weight = torch.as_tensor(model["weight"], dtype=torch.float64)
    bias = torch.as_tensor(model["bias"], dtype=torch.float64)
    normalized = (features - mean) / scale
    return torch.sigmoid(
        torch.einsum("npf,f->np", normalized, weight) + bias
    ).to(dtype=bundle["leaf_probabilities"].dtype)


def relational_hazard_terminal(
    bundle: dict,
    hierarchy,
    model: dict,
) -> torch.Tensor:
    used_hierarchy = bundle.get("_hierarchy", hierarchy)
    hazards = relational_hazard_probabilities(
        bundle, hierarchy, model
    )
    return hierarchical_hazard_terminal_from_hazards(
        bundle["leaf_probabilities"],
        hazards,
        used_hierarchy,
        leaf_nodes=bundle["retained_classes"],
        parent_nodes=bundle["parent_nodes"],
        leaf_node_indices=bundle["leaf_node_indices"],
        parent_node_indices=bundle["parent_node_indices"],
        node_count=int(bundle["node_count"]),
    )


def fit_shared_relational_hazard(
    bundles: list[dict],
    hierarchy,
    *,
    max_iter: int = 150,
    l2_weight: float = 1e-3,
) -> dict:
    """Fit one global linear hazard model from class-disjoint episodes."""
    if not bundles:
        raise ValueError(
            "At least one relational-hazard calibration bundle is required"
        )
    if float(l2_weight) < 0:
        raise ValueError("Relational-hazard L2 weight must be non-negative")
    kinds = [kind for bundle in bundles for kind in bundle["kinds"]]
    groups = [
        group for bundle in bundles for group in bundle["target_groups"]
    ]
    combined_weights = macro_terminal_weights(kinds, groups)
    raw_features = [
        bundle_relational_features(bundle, hierarchy).double()
        for bundle in bundles
    ]
    flattened = torch.cat([
        value.reshape(-1, int(value.shape[-1]))
        for value in raw_features
    ], dim=0)
    feature_mean = flattened.mean(dim=0)
    feature_scale = flattened.std(dim=0, unbiased=False).clamp_min(1e-4)
    prepared = []
    offset = 0
    for bundle, features in zip(bundles, raw_features):
        count = len(bundle["kinds"])
        prepared.append({
            **bundle,
            "_used_hierarchy": bundle.get("_hierarchy", hierarchy),
            "features": (features - feature_mean) / feature_scale,
            "weights": combined_weights[offset:offset + count],
            "leaf_probabilities": bundle[
                "leaf_probabilities"
            ].double(),
            "target_node_indices": bundle[
                "target_node_indices"
            ].long(),
        })
        offset += count

    feature_count = int(flattened.shape[1])
    weight = torch.nn.Parameter(torch.zeros(
        feature_count, dtype=torch.float64
    ))
    bias = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def objective(backward: bool):
        loss = torch.zeros((), dtype=torch.float64)
        for bundle in prepared:
            hazards = torch.sigmoid(
                torch.einsum(
                    "npf,f->np", bundle["features"], weight
                ) + bias
            )
            terminal = hierarchical_hazard_terminal_from_hazards(
                bundle["leaf_probabilities"],
                hazards,
                bundle["_used_hierarchy"],
                leaf_nodes=bundle["retained_classes"],
                parent_nodes=bundle["parent_nodes"],
                leaf_node_indices=bundle["leaf_node_indices"],
                parent_node_indices=bundle["parent_node_indices"],
                node_count=int(bundle["node_count"]),
            )
            probability = terminal.gather(
                1, bundle["target_node_indices"].unsqueeze(1)
            ).squeeze(1)
            loss = loss - (
                bundle["weights"]
                * probability.clamp_min(1e-30).log()
            ).sum()
        loss = loss + float(l2_weight) * weight.square().sum()
        if backward:
            loss.backward()
        return loss

    initial_loss = float(objective(False).detach())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        return objective(True)

    optimizer.step(closure)
    final_loss = float(objective(False).detach())
    state = optimizer.state[weight]
    return {
        "model": "shared_linear_relational_hazard",
        "feature_names": list(RELATIONAL_HAZARD_FEATURE_NAMES),
        "feature_mean": feature_mean.float(),
        "feature_scale": feature_scale.float(),
        "weight": weight.detach().float(),
        "bias": float(bias.detach()),
        "l2_weight": float(l2_weight),
        "initial_objective": initial_loss,
        "final_objective": final_loss,
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
