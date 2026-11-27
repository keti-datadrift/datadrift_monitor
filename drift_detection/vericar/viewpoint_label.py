import json
from pathlib import Path
from tqdm import tqdm


def load_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {Path(line.strip().replace("\\", "/")).name for line in lines if line.strip()}


def main() -> None:
    labels_root = Path(r"F:\IPIU2026\dataset_final\all\labels")
    front_list = Path("txt") / "front_list.txt"
    rear_list = Path("txt") / "rear_list.txt"

    front_set = load_list(front_list)
    rear_set = load_list(rear_list)

    json_paths = list(labels_root.rglob("*.json"))
    for json_path in tqdm(json_paths, desc="Labeling views"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        car = data.get("car")
        if not isinstance(car, dict):
            continue
        image_path = car.get("imagePath")
        if not isinstance(image_path, str):
            continue

        image_key = Path(image_path.replace("\\", "/")).name
        view_value = None
        if image_key in front_set:
            view_value = "front"
        elif image_key in rear_set:
            view_value = "rear"
        else:
            print(f"{image_key} not in front or rear set")

        attributes = car.setdefault("attributes", {})
        if isinstance(attributes, dict):
            attributes["view"] = view_value

        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
