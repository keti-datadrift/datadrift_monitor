from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from torch.nn import functional as F  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.cmpt import (  # noqa: E402
    CMPTCheckpointEvaluator,
    CMPTExperimentSettings,
)
from sacil.cmpt.evaluator import _load_model  # noqa: E402
from sacil.config import load_config_tree  # noqa: E402
from sacil.features import collect_features  # noqa: E402
from sacil.memory import ExemplarMemory  # noqa: E402


# Presentation-oriented supporting text sizes.  The existing main and panel
# title strings and title sizes are intentionally left unchanged.
AXIS_LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize full-training and retained-exemplar NME prototypes "
            "in one shared two-dimensional PCA space."
        )
    )
    parser.add_argument("config", type=Path, help="CMPT evaluation config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--session",
        type=int,
        default=-1,
        help="checkpoint session; -1 selects the final session",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="number of old classes to display",
    )
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "optional original dataset class IDs; otherwise select old "
            "classes with the largest full-vs-memory prototype gap"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def _mirrored_feature_contributions(trainer, model, class_ids, session_id):
    dataset = trainer.data.train_eval_dataset_for_classes(
        class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset,
        shuffle=False,
        session_id=int(session_id) + 27000,
    )
    regular = collect_features(model, loader, trainer.device)
    mirrored = collect_features(
        model,
        loader,
        trainer.device,
        horizontal_flip=True,
    )
    if not torch.equal(regular.indices, mirrored.indices):
        raise RuntimeError("regular and mirrored feature rows are misaligned")
    contributions = 0.5 * (
        F.normalize(regular.features.float(), dim=1)
        + F.normalize(mirrored.features.float(), dim=1)
    )
    return contributions, regular.original_targets, regular.indices


def _prototype_statistics(
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    memory: ExemplarMemory,
    class_ids: tuple[int, ...],
) -> dict[int, dict[str, object]]:
    index_to_position = {
        int(index): position for position, index in enumerate(indices.tolist())
    }
    result: dict[int, dict[str, object]] = {}
    for class_id in class_ids:
        class_mask = targets == int(class_id)
        full_features = features[class_mask]
        memory_indices = memory.indices_for_class(int(class_id))
        if not memory_indices:
            raise ValueError(f"class {class_id} has no retained exemplars")
        try:
            positions = torch.tensor(
                [index_to_position[int(index)] for index in memory_indices],
                dtype=torch.long,
            )
        except KeyError as error:
            raise RuntimeError(
                f"class {class_id} memory index is absent from full data"
            ) from error
        exemplar_features = features[positions]
        full_mean = full_features.mean(dim=0)
        exemplar_mean = exemplar_features.mean(dim=0)
        full_prototype = F.normalize(full_mean, dim=0)
        exemplar_prototype = F.normalize(exemplar_mean, dim=0)
        cosine = float(
            torch.dot(full_prototype, exemplar_prototype).clamp(-1.0, 1.0)
        )
        result[int(class_id)] = {
            "full_features": full_features,
            "exemplar_features": exemplar_features,
            "full_mean": full_mean,
            "exemplar_mean": exemplar_mean,
            "full_prototype": full_prototype,
            "exemplar_prototype": exemplar_prototype,
            "full_count": int(full_features.shape[0]),
            "exemplar_count": int(exemplar_features.shape[0]),
            "cosine_similarity": cosine,
            "cosine_distance": 1.0 - cosine,
            "angle_degrees": math.degrees(math.acos(cosine)),
        }
    return result


def _fit_shared_pca(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = features.mean(dim=0)
    centered = features - center
    _, singular_values, right = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    components = right[:2].T.contiguous()
    explained = singular_values.square()
    explained = explained[:2] / explained.sum().clamp_min(1.0e-12)
    return center, components, explained


def _project(
    values: torch.Tensor,
    center: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    return (values.float() - center) @ components


def _display_name(value: str) -> str:
    return value.replace("_", " ")


def _draw(
    output: Path,
    statistics: dict[int, dict[str, object]],
    selected_classes: list[int],
    class_names: list[str],
    center: torch.Tensor,
    components: torch.Tensor,
    explained: torch.Tensor,
    *,
    learner: str,
    session_id: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    figure = plt.figure(figsize=(17.0, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(1.25, 1.25, 1.0, 1.0))
    overview = figure.add_subplot(grid[:, :2])
    zoom_axes = [
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[0, 3]),
        figure.add_subplot(grid[1, 2]),
        figure.add_subplot(grid[1, 3]),
    ]

    class_handles: list[Line2D] = []
    for offset, class_id in enumerate(selected_classes):
        item = statistics[class_id]
        color = colors[offset % len(colors)]
        name = _display_name(class_names[class_id])
        full_xy = _project(item["full_features"], center, components)
        exemplar_xy = _project(item["exemplar_features"], center, components)
        full_mean_xy = _project(
            item["full_mean"].unsqueeze(0), center, components
        )[0]
        exemplar_mean_xy = _project(
            item["exemplar_mean"].unsqueeze(0), center, components
        )[0]

        overview.scatter(
            full_xy[:, 0],
            full_xy[:, 1],
            s=9,
            alpha=0.12,
            color=color,
            linewidths=0,
            rasterized=True,
        )
        overview.scatter(
            exemplar_xy[:, 0],
            exemplar_xy[:, 1],
            s=30,
            facecolors="none",
            edgecolors=[color],
            linewidths=1.1,
            alpha=0.95,
        )
        overview.scatter(
            full_mean_xy[0],
            full_mean_xy[1],
            marker="*",
            s=260,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=8,
        )
        overview.scatter(
            exemplar_mean_xy[0],
            exemplar_mean_xy[1],
            marker="X",
            s=170,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
        )
        overview.annotate(
            "",
            xy=exemplar_mean_xy.tolist(),
            xytext=full_mean_xy.tolist(),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
            zorder=7,
        )
        overview.annotate(
            f"{name}\nangle={item['angle_degrees']:.2f} deg",
            xy=exemplar_mean_xy.tolist(),
            xytext=(11, 10),
            textcoords="offset points",
            fontsize=ANNOTATION_FONT_SIZE,
            color=color,
            weight="bold",
            zorder=12,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.62,
                "pad": 1.0,
            },
        )
        class_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=color,
                label=f"{name} (ID {class_id})",
                markersize=8,
            )
        )

        axis = zoom_axes[offset]
        axis.scatter(
            full_xy[:, 0],
            full_xy[:, 1],
            s=12,
            alpha=0.18,
            color=color,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            exemplar_xy[:, 0],
            exemplar_xy[:, 1],
            s=38,
            facecolors="none",
            edgecolors=[color],
            linewidths=1.2,
        )
        axis.scatter(
            full_mean_xy[0],
            full_mean_xy[1],
            marker="*",
            s=260,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=8,
        )
        axis.scatter(
            exemplar_mean_xy[0],
            exemplar_mean_xy[1],
            marker="X",
            s=170,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
        )
        axis.annotate(
            "",
            xy=exemplar_mean_xy.tolist(),
            xytext=full_mean_xy.tolist(),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.8},
            zorder=7,
        )
        axis.set_title(
            f"{name} (ID {class_id})\n"
            f"cos={item['cosine_similarity']:.5f}, "
            f"angle={item['angle_degrees']:.2f} deg",
            fontsize=10,
        )
        axis.grid(alpha=0.18)
        axis.set_xlabel("PC1", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.set_ylabel("PC2", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FONT_SIZE,
        )

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            label="Retained exemplar (20)",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Full-data prototype (500)",
            markersize=13,
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Exemplar prototype (20)",
            markersize=10,
        ),
    ]
    overview.legend(
        handles=[*class_handles, *marker_handles],
        loc="best",
        framealpha=0.94,
        fontsize=LEGEND_FONT_SIZE,
    )
    overview.set_title(
        "Shared PCA overview: full training distribution vs retained exemplars",
        fontsize=13,
    )
    overview.set_xlabel(
        f"PC1 ({100.0 * explained[0]:.1f}% variance)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    overview.set_ylabel(
        f"PC2 ({100.0 * explained[1]:.1f}% variance)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    overview.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONT_SIZE,
    )
    overview.grid(alpha=0.18)
    figure.suptitle(
        f"Full-data vs 20-exemplar prototype estimation | "
        f"{learner}, CIFAR-100, session {session_id}",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_config_tree(args.config)
    if args.device is not None:
        config["device"] = args.device
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    evaluator = CMPTCheckpointEvaluator(
        settings,
        PROJECT_ROOT,
        source_root=SOURCE_ROOT,
        progress=print,
    )
    session_id = int(args.session)
    if session_id < 0:
        session_id = len(evaluator.checkpoints) - 1
    if not 0 <= session_id < len(evaluator.checkpoints):
        raise ValueError("requested session is outside the trajectory")
    checkpoint = evaluator.checkpoints[session_id]
    trainer = evaluator._trainer()
    model = _load_model(trainer, checkpoint, session_id)
    trainer.model = model
    trainer.memory = ExemplarMemory.from_state_dict(checkpoint["memory"])

    candidates = trainer.protocol.old_classes(session_id)
    if not candidates:
        candidates = trainer.protocol.seen_classes(session_id)
    contributions, targets, indices = _mirrored_feature_contributions(
        trainer,
        model,
        candidates,
        session_id,
    )
    statistics = _prototype_statistics(
        contributions,
        targets,
        indices,
        trainer.memory,
        candidates,
    )
    if args.class_ids:
        selected_classes = [int(value) for value in args.class_ids]
        missing = [value for value in selected_classes if value not in statistics]
        if missing:
            raise ValueError(f"requested classes are unavailable: {missing}")
    else:
        count = int(args.num_classes)
        if count < 1 or count > len(statistics):
            raise ValueError("num-classes is outside the candidate range")
        selected_classes = sorted(
            statistics,
            key=lambda class_id: float(
                statistics[class_id]["cosine_distance"]
            ),
            reverse=True,
        )[:count]

    pca_features = torch.cat(
        [statistics[value]["full_features"] for value in selected_classes],
        dim=0,
    )
    center, components, explained = _fit_shared_pca(pca_features)
    class_names = list(trainer.data.train_eval.classes)
    output = args.output.expanduser().resolve()
    _draw(
        output,
        statistics,
        selected_classes,
        class_names,
        center,
        components,
        explained,
        learner=settings.learner,
        session_id=session_id,
    )

    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else output.with_suffix(".json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "learner": settings.learner,
        "dataset": "cifar100",
        "session_id": session_id,
        "checkpoint": str(evaluator.checkpoint_paths[session_id]),
        "selection": (
            "explicit_class_ids"
            if args.class_ids
            else "largest_high_dimensional_cosine_distance"
        ),
        "feature_convention": (
            "per-image average of L2-normalized regular and mirrored features"
        ),
        "pca_fit": "all full-training features of displayed classes",
        "pca_explained_variance_ratio": explained.tolist(),
        "marker_note": (
            "pre-L2 class means are plotted; their normalized directions "
            "are the actual NME prototypes"
        ),
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "classes": [],
    }
    candidate_angles = torch.tensor(
        [float(statistics[value]["angle_degrees"]) for value in statistics]
    )
    candidate_cosine_distances = torch.tensor(
        [
            float(statistics[value]["cosine_distance"])
            for value in statistics
        ]
    )
    payload["candidate_pool"] = {
        "old_class_count": len(statistics),
        "angle_degrees": {
            "mean": float(candidate_angles.mean()),
            "median": float(candidate_angles.median()),
            "p75": float(torch.quantile(candidate_angles, 0.75)),
            "p90": float(torch.quantile(candidate_angles, 0.90)),
            "maximum": float(candidate_angles.max()),
        },
        "cosine_distance": {
            "mean": float(candidate_cosine_distances.mean()),
            "median": float(candidate_cosine_distances.median()),
            "p90": float(torch.quantile(candidate_cosine_distances, 0.90)),
            "maximum": float(candidate_cosine_distances.max()),
        },
    }
    for class_id in selected_classes:
        item = statistics[class_id]
        payload["classes"].append(
            {
                "original_class_id": class_id,
                "incremental_class_id": trainer.protocol.incremental_label(
                    class_id
                ),
                "introduction_session": (
                    trainer.protocol.session_for_incremental_label(
                        trainer.protocol.incremental_label(class_id)
                    )
                ),
                "class_name": class_names[class_id],
                "full_count": item["full_count"],
                "exemplar_count": item["exemplar_count"],
                "cosine_similarity": item["cosine_similarity"],
                "cosine_distance": item["cosine_distance"],
                "angle_degrees": item["angle_degrees"],
            }
        )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
