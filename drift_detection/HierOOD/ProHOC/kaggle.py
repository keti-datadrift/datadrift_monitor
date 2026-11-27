import argparse
from pathlib import Path

import kagglehub


COMPETITION = "imagenet-object-localization-challenge"


def download_imagenet(output_dir: str, force_download: bool = False) -> str:
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.competition_download(
        COMPETITION,
        output_dir=str(target_dir),
        force_download=force_download,
    )
    return str(download_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle ImageNet competition files to a target directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the ImageNet competition files will be downloaded.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Redownload files even if they already exist in the output directory.",
    )
    args = parser.parse_args()

    download_path = download_imagenet(
        output_dir=args.output_dir,
        force_download=args.force_download,
    )
    print(f"Downloaded to: {download_path}")


if __name__ == "__main__":
    main()
