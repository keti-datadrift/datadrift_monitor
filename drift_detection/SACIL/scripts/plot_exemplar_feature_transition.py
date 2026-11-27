from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.nn import functional as F


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.anchors import compute_prototypes  # noqa: E402
from sacil.cmpt.evaluator import (  # noqa: E402
    CMPTCheckpointEvaluator,
    CMPTExperimentSettings,
    _full_current_prototypes,
    _full_introduction_prototypes,
    _load_model,
    _paired_support_features,
)
from sacil.config import load_config_tree  # noqa: E402
from sacil.memory import ExemplarMemory  # noqa: E402
from sacil.methods.prototype_transport import (  # noqa: E402
    affine_ridge_transport,
)


DEFAULT_CONFIG = Path(
    "configs/cmpt/common_recipe/evaluate_lucir_natural_affine.yaml"
)
DEFAULT_OUTPUT = Path(
    "iwait2027/figures/"
    "cifar100_lucir_s1_to_s2_exemplar_feature_transformation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the actual movement of paired retained-exemplar "
            "features and CMPT prototypes between two checkpoints."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--previous-session", type=int, default=1)
    parser.add_argument("--current-session", type=int, default=2)
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help=(
            "Incremental class index to plot. By default, select an old "
            "class with clear motion for which CMPT improves prototype "
            "cosine similarity to the full-data oracle."
        ),
    )
    parser.add_argument(
        "--class-count",
        type=int,
        default=5,
        help="Number of well-separated old classes in the multi-class plot.",
    )
    parser.add_argument(
        "--points-per-class",
        type=int,
        default=3,
        help=(
            "Number of retained exemplars displayed per class in the "
            "multi-class plot, taken from the deterministic herding order."
        ),
    )
    parser.add_argument(
        "--class-center-scale",
        type=float,
        default=0.72,
        help=(
            "Display-only scale applied to distances between class centers; "
            "within-class exemplar and prototype movements are preserved."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def _joint_pca(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    values = rows.detach().cpu().float()
    center = values.mean(dim=0, keepdim=True)
    centered = values - center
    _, singular_values, right_h = torch.linalg.svd(
        centered, full_matrices=False
    )
    basis = right_h[:2].T.contiguous()
    projected = centered @ basis
    total_variance = singular_values.square().sum().clamp_min(1.0e-12)
    explained = float(
        (singular_values[:2].square().sum() / total_variance).item()
    )
    return projected, center, basis, explained


def _orient_by_mean_motion(
    projected: torch.Tensor,
    old_count: int,
) -> tuple[torch.Tensor, float]:
    old_mean = projected[:old_count].mean(dim=0)
    current_mean = projected[old_count : 2 * old_count].mean(dim=0)
    displacement = current_mean - old_mean
    norm = float(displacement.norm().item())
    if norm <= 1.0e-12:
        return projected, 0.0
    x_axis = displacement / displacement.norm()
    y_axis = torch.stack([-x_axis[1], x_axis[0]])
    rotation = torch.stack([x_axis, y_axis], dim=1)
    oriented = projected @ rotation
    return oriented, norm


def _plot_transition(
    *,
    old_points: np.ndarray,
    current_points: np.ndarray,
    prototype_points: dict[str, np.ndarray],
    class_name: str,
    previous_session: int,
    current_session: int,
    output: Path,
    dpi: int,
    diagnostic: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.7), constrained_layout=True)
    old_color = "#2F6B9A"
    current_color = "#E67E22"
    arrow_color = "#7A7A7A"
    transformed_color = "#A23B72"
    oracle_color = "#202020"
    nme_color = "#268A62"

    for old, current in zip(old_points, current_points):
        ax.annotate(
            "",
            xy=current,
            xytext=old,
            arrowprops={
                "arrowstyle": "->",
                "color": arrow_color,
                "alpha": 0.34,
                "linewidth": 1.15,
                "shrinkA": 2.5,
                "shrinkB": 2.5,
            },
            zorder=1,
        )

    ax.scatter(
        old_points[:, 0],
        old_points[:, 1],
        s=62,
        marker="o",
        color=old_color,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.92,
        label=rf"Exemplars under $f_{{{previous_session}}}$",
        zorder=3,
    )
    ax.scatter(
        current_points[:, 0],
        current_points[:, 1],
        s=70,
        marker="^",
        color=current_color,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.92,
        label=rf"Same exemplars under $f_{{{current_session}}}$",
        zorder=3,
    )

    previous = prototype_points["previous_stored"]
    transformed = prototype_points["transformed"]
    ax.annotate(
        "",
        xy=transformed,
        xytext=previous,
        arrowprops={
            "arrowstyle": "-|>",
            "color": transformed_color,
            "linewidth": 3.0,
            "mutation_scale": 17,
        },
        zorder=5,
    )
    ax.scatter(
        *previous,
        s=265,
        marker="*",
        color=old_color,
        edgecolors="white",
        linewidths=1.4,
        label=rf"Previous stored prototype $\hat{{\mu}}_{{{previous_session},c}}$",
        zorder=7,
    )
    ax.scatter(
        *transformed,
        s=230,
        marker="X",
        color=transformed_color,
        edgecolors="white",
        linewidths=1.4,
        label=rf"Transformed prototype $\hat{{\mu}}_{{{current_session},c}}$",
        zorder=8,
    )

    if diagnostic:
        oracle = prototype_points["current_full"]
        nme = prototype_points["current_nme"]
        ax.scatter(
            *oracle,
            s=255,
            marker="P",
            color=oracle_color,
            edgecolors="white",
            linewidths=1.4,
            label=rf"Current full-data prototype $\mu^{{F}}_{{{current_session},c}}$",
            zorder=8,
        )
        ax.scatter(
            *nme,
            s=185,
            marker="D",
            color=nme_color,
            edgecolors="white",
            linewidths=1.2,
            label=rf"Current exemplar prototype $\mu^{{E}}_{{{current_session},c}}$",
            zorder=8,
        )

    ax.set_xlabel("Joint PCA dimension 1", fontsize=16)
    ax.set_ylabel("Joint PCA dimension 2", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Actual exemplar-feature transition: {class_name} "
        f"(S{previous_session} to S{current_session})",
        fontsize=17,
        pad=13,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=12,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    ax.margins(0.13)

    suffix = "_diagnostic" if diagnostic else "_paper"
    png_path = output.with_name(output.name + suffix).with_suffix(".png")
    svg_path = output.with_name(output.name + suffix).with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_diagram_inset(
    *,
    old_points: np.ndarray,
    current_points: np.ndarray,
    prototype_points: dict[str, np.ndarray],
    output: Path,
    dpi: int,
) -> None:
    """Save a label-free inset suitable for assembling the method diagram."""

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for old, current in zip(old_points, current_points):
        ax.annotate(
            "",
            xy=current,
            xytext=old,
            arrowprops={
                "arrowstyle": "->",
                "color": "#8A8A8A",
                "alpha": 0.40,
                "linewidth": 1.35,
                "shrinkA": 2.5,
                "shrinkB": 2.5,
            },
            zorder=1,
        )
    ax.scatter(
        old_points[:, 0],
        old_points[:, 1],
        s=75,
        marker="o",
        color="#2F6B9A",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        current_points[:, 0],
        current_points[:, 1],
        s=82,
        marker="^",
        color="#E67E22",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    previous = prototype_points["previous_stored"]
    transformed = prototype_points["transformed"]
    ax.annotate(
        "",
        xy=transformed,
        xytext=previous,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#A23B72",
            "linewidth": 3.5,
            "mutation_scale": 20,
        },
        zorder=5,
    )
    ax.scatter(
        *previous,
        s=300,
        marker="*",
        color="#2F6B9A",
        edgecolors="white",
        linewidths=1.5,
        zorder=7,
    )
    ax.scatter(
        *transformed,
        s=255,
        marker="X",
        color="#A23B72",
        edgecolors="white",
        linewidths=1.5,
        zorder=8,
    )
    ax.margins(0.08)
    ax.axis("off")
    png_path = output.with_name(output.name + "_diagram").with_suffix(".png")
    svg_path = output.with_name(output.name + "_diagram").with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        svg_path,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _select_diverse_classes(
    *,
    priority: torch.Tensor,
    old_features: torch.Tensor,
    current_features: torch.Tensor,
    targets: torch.Tensor,
    class_count: int,
    first_class: int,
) -> list[int]:
    """Greedily balance visible motion with 2D class separation."""

    num_classes = int(priority.numel())
    count = min(int(class_count), num_classes)
    if count <= 0:
        raise ValueError("class count must be positive")
    pooled = torch.cat([old_features.cpu(), current_features.cpu()], dim=0)
    projected, _, _, _ = _joint_pca(pooled)
    exemplar_count = int(old_features.shape[0])
    old_projected = projected[:exemplar_count]
    current_projected = projected[exemplar_count:]
    centroids: list[torch.Tensor] = []
    for class_index in range(num_classes):
        mask = targets.cpu() == class_index
        centroids.append(
            torch.cat(
                [
                    old_projected[mask].mean(dim=0),
                    current_projected[mask].mean(dim=0),
                ]
            )
        )
    centers = torch.stack(centroids)
    scaled_priority = priority.detach().cpu().float()
    scaled_priority = (scaled_priority - scaled_priority.min()) / (
        scaled_priority.max() - scaled_priority.min()
    ).clamp_min(1.0e-12)
    selected = [int(first_class)]
    while len(selected) < count:
        candidate_scores = torch.full((num_classes,), -float("inf"))
        for candidate in range(num_classes):
            if candidate in selected:
                continue
            separation = min(
                float((centers[candidate] - centers[index]).norm().item())
                for index in selected
            )
            candidate_scores[candidate] = (
                0.55 * scaled_priority[candidate] + 0.45 * separation
            )
        selected.append(int(candidate_scores.argmax().item()))
    return selected


def _plot_multi_class_transition(
    *,
    old_points: list[np.ndarray],
    current_points: list[np.ndarray],
    previous_prototypes: np.ndarray,
    transformed_prototypes: np.ndarray,
    class_names: list[str],
    previous_session: int,
    current_session: int,
    output: Path,
    dpi: int,
    diagram: bool,
    class_center_scale: float,
) -> None:
    if not 0.0 < class_center_scale <= 1.0:
        raise ValueError("class center scale must lie in (0, 1]")
    old_points = [values.copy() for values in old_points]
    current_points = [values.copy() for values in current_points]
    previous_prototypes = previous_prototypes.copy()
    transformed_prototypes = transformed_prototypes.copy()
    class_centers = []
    for position in range(len(class_names)):
        class_centers.append(
            np.concatenate(
                [
                    old_points[position],
                    current_points[position],
                    previous_prototypes[position][None, :],
                    transformed_prototypes[position][None, :],
                ],
                axis=0,
            ).mean(axis=0)
        )
    class_centers_array = np.stack(class_centers)
    global_center = class_centers_array.mean(axis=0)
    for position, center in enumerate(class_centers_array):
        target_center = global_center + class_center_scale * (
            center - global_center
        )
        shift = target_center - center
        old_points[position] += shift
        current_points[position] += shift
        previous_prototypes[position] += shift
        transformed_prototypes[position] += shift

    colors = ["#2F6B9A", "#E67E22", "#268A62", "#C44E52", "#8661A8"]
    fig, ax = plt.subplots(
        figsize=((8.4, 6.2) if not diagram else (7.4, 5.2)),
        constrained_layout=True,
    )
    for class_position, (old, current, name) in enumerate(
        zip(old_points, current_points, class_names)
    ):
        color = colors[class_position % len(colors)]
        for old_point, current_point in zip(old, current):
            ax.annotate(
                "",
                xy=current_point,
                xytext=old_point,
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "alpha": 0.58,
                    "linewidth": 1.35,
                    "shrinkA": 2.8,
                    "shrinkB": 2.8,
                },
                zorder=1,
            )
        ax.scatter(
            old[:, 0],
            old[:, 1],
            s=74,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            alpha=0.97,
            zorder=3,
        )
        ax.scatter(
            current[:, 0],
            current[:, 1],
            s=82,
            marker="^",
            color=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.97,
            zorder=3,
        )
        previous = previous_prototypes[class_position]
        transformed = transformed_prototypes[class_position]
        ax.annotate(
            "",
            xy=transformed,
            xytext=previous,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 3.2,
                "mutation_scale": 18,
            },
            zorder=6,
        )
        ax.scatter(
            *previous,
            s=285,
            marker="*",
            facecolors="white",
            edgecolors=color,
            linewidths=2.2,
            zorder=7,
        )
        ax.scatter(
            *transformed,
            s=235,
            marker="X",
            color=color,
            edgecolors="white",
            linewidths=1.3,
            zorder=8,
        )
        if not diagram:
            ax.annotate(
                name,
                xy=transformed,
                xytext=(5, 5),
                textcoords="offset points",
                color=color,
                fontsize=10.5,
                fontweight="bold",
                zorder=9,
            )

    ax.margins(0.09)
    if diagram:
        ax.axis("off")
    else:
        from matplotlib.lines import Line2D

        state_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=8,
                label=rf"Exemplar under $f_{{{previous_session}}}$",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#555555",
                markeredgecolor="white",
                markersize=8,
                label=rf"Same exemplar under $f_{{{current_session}}}$",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=11,
                label="Previous stored prototype",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#555555",
                markeredgecolor="white",
                markersize=9,
                label="Transformed prototype",
            ),
        ]
        ax.legend(
            handles=state_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
            frameon=False,
            fontsize=11.5,
        )
        ax.set_xlabel("Joint PCA dimension 1", fontsize=15)
        ax.set_ylabel("Joint PCA dimension 2", fontsize=15)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.50)
        ax.set_axisbelow(True)
        ax.set_title(
            f"Five-class exemplar and prototype transitions "
            f"(S{previous_session} to S{current_session})",
            fontsize=16.5,
            pad=12,
        )

    suffix = "_five_classes_diagram" if diagram else "_five_classes_paper"
    png_path = output.with_name(output.name + suffix).with_suffix(".png")
    svg_path = output.with_name(output.name + suffix).with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        transparent=diagram,
        facecolor=("none" if diagram else "white"),
        bbox_inches="tight",
        pad_inches=(0.02 if diagram else 0.1),
    )
    fig.savefig(
        svg_path,
        transparent=diagram,
        facecolor=("none" if diagram else "white"),
        bbox_inches="tight",
        pad_inches=(0.02 if diagram else 0.1),
    )
    plt.close(fig)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.current_session != args.previous_session + 1:
        raise ValueError("current session must immediately follow previous session")
    if args.previous_session < 0:
        raise ValueError("previous session must be non-negative")

    config_path = (
        args.config
        if args.config.is_absolute()
        else PROJECT_ROOT / args.config
    ).resolve()
    config = copy.deepcopy(load_config_tree(config_path))
    config["device"] = str(args.device)
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    if settings.transport != "affine_ridge":
        raise ValueError("the visualization requires an affine-ridge config")
    evaluator = CMPTCheckpointEvaluator(
        settings,
        PROJECT_ROOT,
        source_root=SOURCE_ROOT,
        max_sessions=args.current_session + 1,
    )
    trainer = evaluator._trainer()
    checkpoints = evaluator.checkpoints
    models = [
        _load_model(trainer, checkpoint, session_id)
        for session_id, checkpoint in enumerate(checkpoints)
    ]

    bank: torch.Tensor | None = None
    previous_bank: torch.Tensor | None = None
    transition_support = None
    transition_mapping: torch.Tensor | None = None
    transition_residual: float | None = None
    for session_id, (checkpoint, model) in enumerate(zip(checkpoints, models)):
        trainer.model = model
        introduction = _full_introduction_prototypes(
            trainer,
            model,
            session_id,
            horizontal_flip=settings.prototype_horizontal_flip,
        )
        if session_id == 0:
            bank = introduction
            continue
        if bank is None:
            raise RuntimeError("prototype bank was not initialized")
        trainer.memory = ExemplarMemory.from_state_dict(
            checkpoints[session_id - 1]["memory"]
        )
        support = _paired_support_features(
            trainer,
            models[session_id - 1],
            model,
            session_id - 1,
            horizontal_flip=settings.support_horizontal_flip,
        )
        transported_old, mapping, residual = affine_ridge_transport(
            bank,
            support.old_fit_features,
            support.current_fit_features,
            ridge=settings.affine_ridge,
        )
        if session_id == args.current_session:
            previous_bank = bank.detach().cpu().clone()
            transition_support = support
            transition_mapping = mapping.detach().cpu().clone()
            transition_residual = float(residual)
        bank = torch.cat([transported_old.cpu(), introduction.cpu()], dim=0)

    if (
        bank is None
        or previous_bank is None
        or transition_support is None
        or transition_mapping is None
        or transition_residual is None
    ):
        raise RuntimeError("requested transition was not reconstructed")

    old_class_count = trainer.protocol.session(args.current_session).start
    class_ids = trainer.protocol.old_classes(args.current_session)
    previous_full, _ = _full_current_prototypes(
        trainer,
        models[args.previous_session],
        args.previous_session,
        class_ids,
        horizontal_flip=settings.prototype_horizontal_flip,
    )
    current_full, _ = _full_current_prototypes(
        trainer,
        models[args.current_session],
        args.current_session,
        class_ids,
        horizontal_flip=settings.prototype_horizontal_flip,
    )
    incremental_ids = tuple(range(old_class_count))
    previous_nme = compute_prototypes(
        transition_support.old_exemplar_features,
        transition_support.targets,
        incremental_ids,
    ).cpu()
    current_nme = compute_prototypes(
        transition_support.current_exemplar_features,
        transition_support.targets,
        incremental_ids,
    ).cpu()
    current_transformed = bank[:old_class_count].cpu()

    cmpt_similarity = (current_transformed * current_full).sum(dim=1)
    nme_similarity = (current_nme * current_full).sum(dim=1)
    similarity_gain = cmpt_similarity - nme_similarity
    motion = 1.0 - (previous_nme * current_nme).sum(dim=1)
    if args.class_index is None:
        positive = similarity_gain > 0
        score = motion + 2.0 * similarity_gain.clamp_min(0.0)
        if bool(positive.any()):
            score = torch.where(positive, score, torch.full_like(score, -1.0))
        class_index = int(score.argmax().item())
    else:
        class_index = int(args.class_index)
        if not 0 <= class_index < old_class_count:
            raise ValueError(
                f"class index must lie in [0, {old_class_count - 1}]"
            )

    mask = transition_support.targets == class_index
    old_points_high = transition_support.old_exemplar_features[mask].cpu()
    current_points_high = (
        transition_support.current_exemplar_features[mask].cpu()
    )
    prototype_rows = torch.stack(
        [
            previous_bank[class_index],
            current_transformed[class_index],
            previous_full[class_index],
            current_full[class_index],
            previous_nme[class_index],
            current_nme[class_index],
        ],
        dim=0,
    )
    joint_rows = torch.cat(
        [old_points_high, current_points_high, prototype_rows], dim=0
    )
    projected, pca_center, pca_basis, explained = _joint_pca(joint_rows)
    projected, oriented_motion = _orient_by_mean_motion(
        projected, old_points_high.shape[0]
    )
    count = old_points_high.shape[0]
    old_points = projected[:count].numpy()
    current_points = projected[count : 2 * count].numpy()
    prototype_projected = projected[2 * count :].numpy()
    prototype_points = {
        "previous_stored": prototype_projected[0],
        "transformed": prototype_projected[1],
        "previous_full": prototype_projected[2],
        "current_full": prototype_projected[3],
        "previous_nme": prototype_projected[4],
        "current_nme": prototype_projected[5],
    }

    original_class_id = int(class_ids[class_index])
    class_name = str(trainer.data.train_eval.classes[original_class_id])
    output = (
        args.output
        if args.output.is_absolute()
        else PROJECT_ROOT / args.output
    ).resolve()
    for diagnostic in (False, True):
        _plot_transition(
            old_points=old_points,
            current_points=current_points,
            prototype_points=prototype_points,
            class_name=class_name,
            previous_session=args.previous_session,
            current_session=args.current_session,
            output=output,
            dpi=args.dpi,
            diagnostic=diagnostic,
        )
    _plot_diagram_inset(
        old_points=old_points,
        current_points=current_points,
        prototype_points=prototype_points,
        output=output,
        dpi=args.dpi,
    )

    selection_priority = motion + 2.0 * similarity_gain.clamp_min(0.0)
    selected_classes = _select_diverse_classes(
        priority=selection_priority,
        old_features=transition_support.old_exemplar_features,
        current_features=transition_support.current_exemplar_features,
        targets=transition_support.targets,
        class_count=args.class_count,
        first_class=class_index,
    )
    multi_old_high: list[torch.Tensor] = []
    multi_current_high: list[torch.Tensor] = []
    multi_counts: list[int] = []
    multi_names: list[str] = []
    for selected in selected_classes:
        selected_mask = transition_support.targets == selected
        old_selected = transition_support.old_exemplar_features[
            selected_mask
        ].cpu()
        current_selected = transition_support.current_exemplar_features[
            selected_mask
        ].cpu()
        displayed_count = min(int(args.points_per_class), len(old_selected))
        if displayed_count <= 0:
            raise ValueError("points per class must be positive")
        old_selected = old_selected[:displayed_count]
        current_selected = current_selected[:displayed_count]
        multi_old_high.append(old_selected)
        multi_current_high.append(current_selected)
        multi_counts.append(int(old_selected.shape[0]))
        selected_original_id = int(class_ids[selected])
        multi_names.append(
            str(trainer.data.train_eval.classes[selected_original_id])
        )
    multi_old_rows = torch.cat(multi_old_high, dim=0)
    multi_current_rows = torch.cat(multi_current_high, dim=0)
    multi_prototype_rows = torch.cat(
        [
            previous_bank[selected_classes],
            current_transformed[selected_classes],
        ],
        dim=0,
    )
    multi_joint_rows = torch.cat(
        [multi_old_rows, multi_current_rows, multi_prototype_rows], dim=0
    )
    (
        multi_projected,
        multi_pca_center,
        multi_pca_basis,
        multi_explained,
    ) = _joint_pca(multi_joint_rows)
    multi_projected, multi_oriented_motion = _orient_by_mean_motion(
        multi_projected, int(multi_old_rows.shape[0])
    )
    multi_total = int(multi_old_rows.shape[0])
    multi_old_projected = multi_projected[:multi_total].numpy()
    multi_current_projected = multi_projected[
        multi_total : 2 * multi_total
    ].numpy()
    multi_prototype_projected = multi_projected[2 * multi_total :].numpy()
    multi_old_points: list[np.ndarray] = []
    multi_current_points: list[np.ndarray] = []
    cursor = 0
    for selected_count in multi_counts:
        stop = cursor + selected_count
        multi_old_points.append(multi_old_projected[cursor:stop])
        multi_current_points.append(multi_current_projected[cursor:stop])
        cursor = stop
    for diagram in (False, True):
        _plot_multi_class_transition(
            old_points=multi_old_points,
            current_points=multi_current_points,
            previous_prototypes=multi_prototype_projected[
                : len(selected_classes)
            ],
            transformed_prototypes=multi_prototype_projected[
                len(selected_classes) :
            ],
            class_names=multi_names,
            previous_session=args.previous_session,
            current_session=args.current_session,
            output=output,
            dpi=args.dpi,
            diagram=diagram,
            class_center_scale=float(args.class_center_scale),
        )

    payload = {
        "config": str(config_path),
        "learner": settings.learner,
        "previous_session": int(args.previous_session),
        "current_session": int(args.current_session),
        "incremental_class_index": class_index,
        "original_class_id": original_class_id,
        "class_name": class_name,
        "exemplar_count": int(count),
        "affine_ridge": float(settings.affine_ridge),
        "fit_support_count": int(transition_support.fit_support_count),
        "fit_residual": transition_residual,
        "cmpt_to_oracle_cosine_similarity": float(
            cmpt_similarity[class_index].item()
        ),
        "nme_to_oracle_cosine_similarity": float(
            nme_similarity[class_index].item()
        ),
        "cmpt_cosine_similarity_gain": float(
            similarity_gain[class_index].item()
        ),
        "exemplar_prototype_cosine_motion": float(motion[class_index].item()),
        "joint_pca_explained_variance_ratio": explained,
        "oriented_2d_mean_motion_before_rotation": oriented_motion,
        "multi_class_visualization": {
            "incremental_class_indices": selected_classes,
            "original_class_ids": [
                int(class_ids[index]) for index in selected_classes
            ],
            "class_names": multi_names,
            "exemplars_per_class": multi_counts,
            "display_only_class_center_scale": float(
                args.class_center_scale
            ),
            "joint_pca_explained_variance_ratio": multi_explained,
            "oriented_2d_mean_motion_before_rotation": (
                multi_oriented_motion
            ),
            "projection_center": multi_pca_center.flatten().tolist(),
            "projection_basis": multi_pca_basis.tolist(),
        },
        "projection": {
            "fit": "one PCA basis fitted jointly to old/current points and prototypes",
            "orientation": "2D axes rigidly rotated so mean exemplar motion points right",
            "center": pca_center.flatten().tolist(),
            "basis": pca_basis.tolist(),
        },
        "files": {
            "paper_png": str(
                output.with_name(output.name + "_paper").with_suffix(".png")
            ),
            "paper_svg": str(
                output.with_name(output.name + "_paper").with_suffix(".svg")
            ),
            "diagnostic_png": str(
                output.with_name(output.name + "_diagnostic").with_suffix(".png")
            ),
            "diagnostic_svg": str(
                output.with_name(output.name + "_diagnostic").with_suffix(".svg")
            ),
            "diagram_png": str(
                output.with_name(output.name + "_diagram").with_suffix(".png")
            ),
            "diagram_svg": str(
                output.with_name(output.name + "_diagram").with_suffix(".svg")
            ),
            "five_classes_paper_png": str(
                output.with_name(output.name + "_five_classes_paper").with_suffix(".png")
            ),
            "five_classes_paper_svg": str(
                output.with_name(output.name + "_five_classes_paper").with_suffix(".svg")
            ),
            "five_classes_diagram_png": str(
                output.with_name(output.name + "_five_classes_diagram").with_suffix(".png")
            ),
            "five_classes_diagram_svg": str(
                output.with_name(output.name + "_five_classes_diagram").with_suffix(".svg")
            ),
        },
    }
    metadata_path = output.with_name(output.name + "_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
