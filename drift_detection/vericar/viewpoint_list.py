import os
from pathlib import Path

def main():
    data_root = Path(r"E:\viewpoint")
    front_dir = data_root / "front"
    rear_dir = data_root / "rear"

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    output_dir = Path("txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    front_txt = output_dir / "front_list.txt"
    rear_txt = output_dir / "rear_list.txt"

    def collect_files(root: Path) -> list[Path]:
        files: list[Path] = []
        for ext in extensions:
            files.extend(root.rglob(ext))
        return sorted(set(files))

    front_files = collect_files(front_dir)
    rear_files = collect_files(rear_dir)

    front_txt.write_text(
        "\n".join(str(p.relative_to(front_dir)) for p in front_files),
        encoding="utf-8",
    )
    rear_txt.write_text(
        "\n".join(str(p.relative_to(rear_dir)) for p in rear_files),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
