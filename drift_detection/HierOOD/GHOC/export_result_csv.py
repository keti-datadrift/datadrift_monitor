from __future__ import annotations

import argparse
from pathlib import Path

from core.result_export import export_result_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=False)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "csv"
    export_result_to_csv(input_path, output_dir)


if __name__ == "__main__":
    main()
