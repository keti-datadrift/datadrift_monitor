import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from analyze.depthwise_temperature_experiment import (
    plot_best_comparison,
    plot_best_depth_profiles,
    summarize_best,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--num_shards", required=True, type=int)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    shard_paths = [
        os.path.join(input_dir, f"depthwise_temperature_search.shard_{idx}.csv")
        for idx in range(args.num_shards)
    ]
    missing = [path for path in shard_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing shard result files:\n" + "\n".join(missing)
        )

    global_path = os.path.join(input_dir, "global_temperature_sweep.csv")
    if not os.path.exists(global_path):
        raise FileNotFoundError(f"Missing global sweep CSV: {global_path}")

    global_rows = pd.read_csv(global_path).to_dict("records")
    depth_rows = []
    for path in shard_paths:
        depth_rows.extend(pd.read_csv(path).to_dict("records"))

    depth_rows = sorted(depth_rows, key=lambda row: (row["run"], row["temperature_spec"]))
    write_csv(os.path.join(input_dir, "depthwise_temperature_search.csv"), depth_rows)

    summary_rows = summarize_best(global_rows, depth_rows)
    write_csv(os.path.join(input_dir, "temperature_comparison_summary.csv"), summary_rows)
    plot_best_comparison(summary_rows, os.path.join(input_dir, "global_vs_depthwise_best.png"))
    plot_best_depth_profiles(summary_rows, os.path.join(input_dir, "best_depthwise_temperature_profiles.png"))

    print(f"Merged {args.num_shards} shard files in {input_dir}")


if __name__ == "__main__":
    main()
