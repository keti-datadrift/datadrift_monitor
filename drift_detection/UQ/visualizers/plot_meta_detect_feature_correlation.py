import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

META_DETECT_RUN_PATH = (
    r"object_detectors/runs/predict/coco/00-00-0000_00;00_meta_detect"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
FIGSIZE_SCALE = 0.32
MIN_FIGSIZE = 10.0

METADATA_COLUMNS = {
    "image_id",
    "image_path",
    "pred_idx",
    "raw_pred_idx",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
    "pred_class",
}

CLASS_PROB_VECTOR_NAME = "class_probability_vector"

SEMANTIC_GROUP_ORDER = [
    "class_probability",
    "score",
    "localization",
    "other",
]

SEMANTIC_GROUP_LABELS = {
    "class_probability": "Class probability",
    "score": "Score",
    "localization": "Localization",
    "other": "Other",
}

SOURCE_GROUP_ORDER = [
    "final_prediction_only",
    "candidate_required",
    "other",
]

SOURCE_GROUP_LABELS = {
    "final_prediction_only": "Final prediction only",
    "candidate_required": "Candidate boxes required",
    "other": "Other",
}


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def resolve_meta_detect_csv(raw_path: str) -> Path:
    path = resolve_repo_path(raw_path)
    if path.is_file():
        return path
    csv_path = path / "meta_detect.csv"
    if csv_path.is_file():
        return csv_path
    raise FileNotFoundError(f"meta_detect.csv not found from path: {path}")


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    if not columns:
        raise ValueError("No numeric meta_detect feature columns found.")
    return columns


def collapse_class_probability_vector(columns: list[str]) -> list[dict[str, object]]:
    prob_cols = sorted(
        [col for col in columns if col.startswith("prob_") and col[5:].isdigit()],
        key=lambda col: int(col[5:]),
    )
    prob_set = set(prob_cols)
    features: list[dict[str, object]] = []
    inserted_prob_vector = False
    for col in columns:
        if col in prob_set:
            if not inserted_prob_vector:
                features.append({"name": CLASS_PROB_VECTOR_NAME, "columns": prob_cols})
                inserted_prob_vector = True
            continue
        features.append({"name": col, "columns": [col]})
    return features


def semantic_group(feature_name: str) -> str:
    if feature_name in {"prob_sum", CLASS_PROB_VECTOR_NAME}:
        return "class_probability"
    if feature_name.startswith("score_"):
        return "score"
    if feature_name in {"size", "circum", "size_circum"} or feature_name.startswith(
        ("x_", "y_", "w_", "h_", "size_", "circum_", "size_circum_", "iou_pb_")
    ):
        return "localization"
    return "other"


def source_group(feature_name: str) -> str:
    if (
        feature_name == "prob_sum"
        or feature_name == CLASS_PROB_VECTOR_NAME
        or feature_name in {"size", "circum", "size_circum"}
    ):
        return "final_prediction_only"
    if feature_name.startswith(
        (
            "x_",
            "y_",
            "w_",
            "h_",
            "size_",
            "circum_",
            "size_circum_",
            "score_",
            "iou_pb_",
        )
    ):
        return "candidate_required"
    return "other"


def prob_sort_key(feature_name: str) -> tuple[int, int, str]:
    if feature_name == "prob_sum":
        return (0, -1, feature_name)
    if feature_name == CLASS_PROB_VECTOR_NAME:
        return (0, 0, feature_name)
    return (1, 0, feature_name)


def sort_features(
    features: list[dict[str, object]], group_fn, group_order: list[str]
) -> tuple[list[dict[str, object]], dict[str, str]]:
    group_index = {name: idx for idx, name in enumerate(group_order)}
    group_by_feature = {
        str(feature["name"]): group_fn(str(feature["name"])) for feature in features
    }

    def key(feature: dict[str, object]):
        name = str(feature["name"])
        group = group_by_feature[name]
        return (group_index.get(group, len(group_order)), prob_sort_key(name), name)

    return sorted(features, key=key), group_by_feature


def correlation_matrix(
    df: pd.DataFrame, features: list[dict[str, object]]
) -> pd.DataFrame:
    all_columns = []
    for feature in features:
        all_columns.extend(str(col) for col in feature["columns"])
    all_columns = list(dict.fromkeys(all_columns))
    base_corr = (
        df[all_columns].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    )
    base_corr = base_corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    names = [str(feature["name"]) for feature in features]
    matrix = np.zeros((len(features), len(features)), dtype=float)
    for row_idx, row_feature in enumerate(features):
        row_cols = [str(col) for col in row_feature["columns"]]
        for col_idx, col_feature in enumerate(features):
            if row_idx == col_idx:
                matrix[row_idx, col_idx] = 1.0
                continue
            col_cols = [str(col) for col in col_feature["columns"]]
            sub_corr = base_corr.loc[row_cols, col_cols].to_numpy(dtype=float)
            matrix[row_idx, col_idx] = (
                float(np.nanmean(sub_corr)) if sub_corr.size else 0.0
            )
    return pd.DataFrame(matrix, index=names, columns=names)


def group_boundaries(
    columns: list[str], group_by_feature: dict[str, str]
) -> list[tuple[str, int, int]]:
    if not columns:
        return []
    spans = []
    start = 0
    current = group_by_feature.get(columns[0], "other")
    for idx, col in enumerate(columns[1:], start=1):
        group = group_by_feature.get(col, "other")
        if group != current:
            spans.append((current, start, idx))
            start = idx
            current = group
    spans.append((current, start, len(columns)))
    return spans


def plot_corr(
    corr: pd.DataFrame,
    group_by_feature: dict[str, str],
    group_labels: dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(corr.columns)
    size = max(MIN_FIGSIZE, len(columns) * FIGSIZE_SCALE)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        corr.to_numpy(dtype=float),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticklabels(columns, fontsize=7)

    for group, start, end in group_boundaries(columns, group_by_feature):
        label = group_labels.get(group, group)
        center = (start + end - 1) / 2.0
        ax.axhline(start - 0.5, color="black", linewidth=0.8)
        ax.axvline(start - 0.5, color="black", linewidth=0.8)
        ax.text(
            -1.8, center, label, va="center", ha="right", fontsize=8, fontweight="bold"
        )
        ax.text(
            center,
            -1.8,
            label,
            va="bottom",
            ha="center",
            rotation=90,
            fontsize=8,
            fontweight="bold",
        )
    ax.axhline(len(columns) - 0.5, color="black", linewidth=0.8)
    ax.axvline(len(columns) - 0.5, color="black", linewidth=0.8)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def make_output_dir(meta_detect_csv: Path) -> Path:
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    run_name = (
        meta_detect_csv.parent.name
        if meta_detect_csv.name == "meta_detect.csv"
        else meta_detect_csv.stem
    )
    safe_run_name = "".join(
        ch if ch.isalnum() or ch in {"_", "-", ";"} else "_" for ch in run_name
    ).strip("_")
    return OUTPUT_ROOT / f"{timestamp}_meta_detect_feature_correlation_{safe_run_name}"


def write_group_csv(
    columns: list[str], group_by_feature: dict[str, str], out_path: Path
) -> None:
    rows = [
        {"feature": col, "group": group_by_feature.get(col, "other")} for col in columns
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-detect-root",
        type=str,
        default=META_DETECT_RUN_PATH,
        help="Path to a meta_detect run directory or meta_detect.csv.",
    )
    args = parser.parse_args()

    meta_detect_csv = resolve_meta_detect_csv(args.meta_detect_root)
    df = pd.read_csv(meta_detect_csv)
    columns = feature_columns(df)
    features = collapse_class_probability_vector(columns)
    out_dir = make_output_dir(meta_detect_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_features, semantic_groups = sort_features(
        features, semantic_group, SEMANTIC_GROUP_ORDER
    )
    semantic_columns = [str(feature["name"]) for feature in semantic_features]
    semantic_corr = correlation_matrix(df, semantic_features)
    semantic_corr.to_csv(out_dir / "semantic_group_correlation.csv")
    write_group_csv(
        semantic_columns, semantic_groups, out_dir / "semantic_group_features.csv"
    )
    plot_corr(
        semantic_corr,
        semantic_groups,
        SEMANTIC_GROUP_LABELS,
        "MetaDetect Feature Correlation by Semantic Group",
        out_dir / "semantic_group_correlation_matrix.png",
    )

    source_features, source_groups = sort_features(
        features, source_group, SOURCE_GROUP_ORDER
    )
    source_columns = [str(feature["name"]) for feature in source_features]
    source_corr = correlation_matrix(df, source_features)
    source_corr.to_csv(out_dir / "source_group_correlation.csv")
    write_group_csv(
        source_columns, source_groups, out_dir / "source_group_features.csv"
    )
    plot_corr(
        source_corr,
        source_groups,
        SOURCE_GROUP_LABELS,
        "MetaDetect Feature Correlation by Feature Source",
        out_dir / "source_group_correlation_matrix.png",
    )

    metadata = {
        "meta_detect_csv": str(meta_detect_csv),
        "num_rows": int(len(df)),
        "num_features": int(len(columns)),
        "num_display_features": int(len(features)),
        "class_probability_vector_columns": [
            str(col)
            for feature in features
            if feature["name"] == CLASS_PROB_VECTOR_NAME
            for col in feature["columns"]
        ],
        "outputs": [
            "semantic_group_correlation_matrix.png",
            "source_group_correlation_matrix.png",
        ],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved MetaDetect correlation plots: {out_dir}")


if __name__ == "__main__":
    main()
