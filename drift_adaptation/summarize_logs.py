from pathlib import Path
import math

from tensorboard.backend.event_processing import event_accumulator as ea

# ====================================
METHOD_TO_VERSION = {
    "no" : r"./runs/drift_adaptation/v157",
    "n1" : r"./runs/drift_adaptation/v158",
    "n2" : r"./runs/drift_adaptation/v159",
    "gmm": r"./runs/drift_adaptation/v160",
}
# ====================================

ADAPT_TAGS = [
    "Adaptation_epoch/total",
    "Adaptation_metrics/precision",
    "Adaptation_metrics/recall",
    "Adaptation_metrics/mAP50",
    "Adaptation_metrics/mAP50-95",
]
RETN_TAGS = [
    "Retention_epoch/total",
    "Retention_metrics/precision",
    "Retention_metrics/recall",
    "Retention_metrics/mAP50",
    "Retention_metrics/mAP50-95",
]
ALL_TAGS = ADAPT_TAGS + RETN_TAGS


def load_scalars_from_logs(logs_dir: Path):
    acc = ea.EventAccumulator(str(logs_dir), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    avail = set(acc.Tags().get("scalars", []))
    out = {}
    for tag in ALL_TAGS:
        if tag in avail:
            out[tag] = [ev.value for ev in acc.Scalars(tag)]
        else:
            out[tag] = []
    return out


def mean_or_nan(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def main():
    rows = []
    for method, ver_path in METHOD_TO_VERSION.items():
        p = Path(ver_path).expanduser().resolve()
        logs_dir = p / "logs"
        if not logs_dir.exists():
            print(f"[WARN] logs 디렉토리 없음: {logs_dir}  (method={method})")
            vals = {tag: float("nan") for tag in ALL_TAGS}
        else:
            scalars = load_scalars_from_logs(logs_dir)
            vals = {tag: mean_or_nan(scalars.get(tag, [])) for tag in ALL_TAGS}
        rows.append((method, vals))

    col_names = ["method"] + ALL_TAGS
    col_widths = [6] + [10] * len(ALL_TAGS)

    def fmt_cell(val, w):
        if isinstance(val, float):
            if math.isnan(val):
                return "NaN".rjust(w)
            return f"{val:.6f}".rjust(w)
        return str(val).rjust(w)

    header = " | ".join(name.rjust(w) for name, w in zip(col_names, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    print(header)
    print(sep)

    for method, vals in rows:
        line_cells = [method] + [vals[tag] for tag in ALL_TAGS]
        print(" | ".join(fmt_cell(c, w) for c, w in zip(line_cells, col_widths)))

    print()


if __name__ == "__main__":
    main()
