import argparse
import base64
import io
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from model import VehiInfoRet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TENSOR_EXTS = {".pt", ".pth", ".npy"}


def _parse_label_file(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("label", "class", "class_id", "name"):
                if key in data:
                    return str(data[key])
            car = data.get("car")
            if isinstance(car, dict):
                attributes = car.get("attributes", {})
                if isinstance(attributes, dict):
                    brand = attributes.get("brand", "")
                    model = attributes.get("model", "")
                    year = attributes.get("year", "")
                    car_type = attributes.get("type", "")
                    parts = [p for p in (car_type, brand, model, year) if p]
                    if parts:
                        return "/".join(parts)
        return json.dumps(data, ensure_ascii=False)
    return "unknown"


def _build_label_index(labels_dir: Path) -> tuple[dict, dict]:
    index: dict = {}
    preview_index: dict = {}
    label_paths = [p for p in labels_dir.rglob("*")]
    for label_path in tqdm(label_paths, desc="Indexing labels"):
        if label_path.suffix.lower() not in {".json", ".txt"}:
            continue
        label = _parse_label_file(label_path)
        index[label_path.stem] = label
        if label_path.suffix.lower() == ".json":
            try:
                data = json.loads(label_path.read_text(encoding="utf-8"))
                car = data.get("car", {})
                image_path = car.get("imagePath")
                if image_path:
                    index[Path(str(image_path)).stem] = label
                    preview_index[Path(str(image_path)).stem] = str(image_path)
            except Exception:
                pass
    return index, preview_index


def _load_label_map(label_map_path: Path) -> dict:
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    mapping: dict = {}
    if isinstance(data, dict):
        for k, v in data.items():
            mapping[Path(k).stem] = str(v)
        return mapping
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                image_key = None
                for key in ("image", "filename", "file", "path"):
                    if key in item:
                        image_key = item[key]
                        break
                if image_key is None:
                    continue
                label_val = None
                for key in ("label", "class", "class_id", "name"):
                    if key in item:
                        label_val = item[key]
                        break
                if label_val is None:
                    continue
                mapping[Path(str(image_key)).stem] = str(label_val)
        return mapping
    raise ValueError("Unsupported label_map JSON format")


def _collect_images_and_labels(
    data_path: Path, label_map: dict | None
) -> List[Tuple[Path, str]]:
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images folder not found: {images_dir}")
    if label_map is None and not labels_dir.exists():
        raise FileNotFoundError(f"labels folder not found: {labels_dir}")

    label_index = None
    if label_map is None:
        label_index, _ = _build_label_index(labels_dir)

    items: List[Tuple[Path, str]] = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in TENSOR_EXTS:
            continue
        label = "unknown"
        if label_map and img_path.stem in label_map:
            label = label_map[img_path.stem]
        elif label_index and img_path.stem in label_index:
            label = label_index[img_path.stem]
        items.append((img_path, label))

    if not items:
        raise RuntimeError(f"No images found in {images_dir}")
    return items


def _encode_thumbnail(image_path: Path, max_size: int = 256) -> str:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _tensor_to_thumbnail(
    tensor: torch.Tensor,
    max_size: int = 256,
    mean: List[float] | None = None,
    std: List[float] | None = None,
) -> str:
    if tensor.ndim != 3:
        raise ValueError("Expected tensor with shape [C,H,W] or [H,W,C] for preview.")
    image = tensor.detach().cpu().float()
    if image.shape[0] != 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if mean is not None and std is not None and len(mean) == 3 and len(std) == 3:
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        image = image * std_t + mean_t
    if image.max() > 1.0 or image.min() < 0.0:
        if image.min() >= -1.0 and image.max() <= 1.0:
            image = (image + 1.0) / 2.0
        else:
            image = (image / 255.0)
    image = image.clamp(0, 1)
    image = (image * 255.0).byte()
    image = image.permute(1, 2, 0).numpy()
    pil = Image.fromarray(image)
    pil.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _load_preprocessed_tensor(path: Path) -> torch.Tensor:
    ext = path.suffix.lower()
    if ext in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {path}")
        return tensor
    if ext == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr)
    raise ValueError(f"Unsupported preprocessed tensor: {path}")


def _extract_features(
    model: VehiInfoRet,
    items: List[Tuple[Path, str]],
    batch_size: int,
    device: torch.device,
    preview_from_tensor: bool,
    preview_mean: List[float] | None,
    preview_std: List[float] | None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    model.to(device)
    model.eval()

    features_list = []
    labels = []
    img_paths = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), batch_size), desc="Extracting features"):
            batch_items = items[i : i + batch_size]
            tensors = []
            for path, _ in batch_items:
                if path.suffix.lower() in IMAGE_EXTS:
                    raise RuntimeError(
                        f"Found raw image {path}. Remove it or pre-save tensors "
                        f"(.pt/.pth/.npy) since preprocessing is disabled."
                    )
                tensor = _load_preprocessed_tensor(path)
                tensors.append(tensor)
            batch = torch.stack(tensors, dim=0).to(device)
            feats = model(batch).cpu().numpy()
            features_list.append(feats)
            labels.extend([label for _, label in batch_items])
            if preview_from_tensor:
                for t in tensors:
                    img_paths.append(
                        _tensor_to_thumbnail(
                            t, mean=preview_mean, std=preview_std
                        )
                    )
            else:
                img_paths.extend([str(path) for path, _ in batch_items])

    features = np.concatenate(features_list, axis=0)
    return features, labels, img_paths


def _tsne_3d(features: np.ndarray, seed: int = 123) -> np.ndarray:
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for 3D t-SNE.")
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
        early_exaggeration=50.0,
    )
    print("Running t-SNE (this can take a while)...")
    return tsne.fit_transform(features)


def _write_html(
    coords: np.ndarray,
    labels: List[str],
    img_paths: List[str],
    img_names: List[str],
    output_html: Path,
    use_thumbnails: bool,
):
    if use_thumbnails:
        img_sources = []
        for p in img_paths:
            if not p:
                img_sources.append("")
            elif p.startswith("data:image/"):
                img_sources.append(p)
            else:
                img_sources.append(_encode_thumbnail(Path(p)))
    else:
        img_sources = []
        for p in img_paths:
            if not p:
                img_sources.append("")
            elif p.startswith("data:image/"):
                img_sources.append(p)
            else:
                img_sources.append(Path(p).resolve().as_uri())

    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    z = coords[:, 2].tolist()

    type_hues = {
        "해치백": 0,
        "hatchback": 0,
        "Hatchback": 0,
        "SUV": 210,
        "suv": 210,
        "세단": 120,
        "sedan": 120,
        "Sedan": 120,
        "화물": 0,
        "truck": 0,
        "Truck": 0,
        "승합": 320,
        "van": 320,
        "Van": 320,
        "unknown": 0,
    }

    def label_type(label: str) -> str:
        if not label or label == "unknown":
            return "unknown"
        parts = label.split("/")
        return parts[0] if parts else "unknown"

    type_to_labels: dict = {}
    for label in labels:
        t = label_type(label)
        type_to_labels.setdefault(t, []).append(label)

    label_to_color = {}
    for t, t_labels in type_to_labels.items():
        hue = type_hues.get(t, 200)
        unique = []
        for l in t_labels:
            if l not in unique:
                unique.append(l)
        for idx, l in enumerate(unique):
            light = 35 + (idx % 6) * 8
            sat = 70
            if t in {"화물", "truck", "Truck"}:
                sat = 10
                light = 35 + (idx % 6) * 6
            label_to_color[l] = f"hsl({hue}, {sat}%, {light}%)"

    colors = [label_to_color.get(l, "hsl(0, 0%, 60%)") for l in labels]

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>3D t-SNE Feature Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
      body {{ margin: 0; font-family: Arial, sans-serif; }}
      #container {{ display: grid; grid-template-columns: 1fr 320px; height: 100vh; }}
      #plot {{ width: 100%; height: 100%; }}
      #sidebar {{ padding: 12px; border-left: 1px solid #ddd; }}
      #preview {{ width: 100%; height: auto; border: 1px solid #ddd; }}
      #label {{ margin-top: 8px; font-size: 14px; }}
    </style>
  </head>
  <body>
    <div id="container">
      <div id="plot"></div>
      <div id="sidebar">
        <img id="preview" src="" alt="Click a point">
        <div id="label">Click a point to see image/label.</div>
      </div>
    </div>
    <script>
      const x = {json.dumps(x)};
      const y = {json.dumps(y)};
      const z = {json.dumps(z)};
      const labels = {json.dumps(labels)};
      const imgNames = {json.dumps(img_names)};
      const colors = {json.dumps(colors)};
      const imgSources = {json.dumps(img_sources)};

      const trace = {{
        x, y, z,
        mode: "markers",
        type: "scatter3d",
        marker: {{
          size: 3,
          color: colors,
          opacity: 0.85
        }},
        text: labels,
        hovertext: imgNames,
        customdata: imgSources,
        hovertemplate: "Class: %{{text}}<br>Image: %{{hovertext}}<extra></extra>"
      }};

      const layout = {{
        margin: {{ l: 0, r: 0, t: 0, b: 0 }},
        scene: {{
          xaxis: {{ visible: false }},
          yaxis: {{ visible: false }},
          zaxis: {{ visible: false }}
        }}
      }};

      Plotly.newPlot("plot", [trace], layout);

      const plot = document.getElementById("plot");
      const preview = document.getElementById("preview");
      const label = document.getElementById("label");
      plot.on("plotly_click", (data) => {{
        const idx = data.points[0].pointIndex;
        const src = imgSources[idx];
        if (src) {{
          preview.src = src;
        }} else {{
          preview.removeAttribute("src");
        }}
        label.textContent = labels[idx] || "unknown";
      }});
    </script>
  </body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def _find_preview_path(preview_dir: Path, stem: str) -> str:
    for ext in IMAGE_EXTS:
        candidate = preview_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    for candidate in preview_dir.rglob(f"{stem}.*"):
        if candidate.suffix.lower() in IMAGE_EXTS:
            return str(candidate)
    return ""


def _build_image_index(images_root: Path) -> dict:
    index: dict = {}
    image_paths = [p for p in images_root.rglob("*")]
    for path in tqdm(image_paths, desc="Indexing images"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        index.setdefault(path.stem, str(path))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path(r"F:\IPIU2026\logs\mining\vericar_experiment_seed1234\version_1-HiMSwei\checkpoints\best-loss-epoch=53-val_loss=0.6140.ckpt"),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(r"F:\IPIU2026\dataset_final\step0\train"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output_html", type=Path, default=Path("tsne_feature_view.html")
    )
    parser.add_argument(
        "--features_cache",
        type=Path,
        default=Path("cached_features.pt"),
        help="Path to cached features (.pt). If exists, skip extraction.",
    )
    parser.add_argument(
        "--preview_images_dir",
        type=Path,
        default=None,
        help="Directory with raw images for preview; matched by stem.",
    )
    parser.add_argument(
        "--preview_base_dir",
        type=Path,
        default=None,
        help="Base directory for imagePath entries inside label JSON.",
    )
    parser.add_argument(
        "--label_map_json",
        type=Path,
        default=None,
        help="Optional JSON file mapping image filename to label.",
    )
    parser.add_argument(
        "--use_thumbnails",
        action="store_true",
        help="Embed thumbnails as base64 for easier local viewing.",
    )
    parser.add_argument(
        "--preview_from_tensor",
        action="store_true",
        help="Generate preview images from preprocessed tensors (.pt/.npy).",
    )
    parser.add_argument(
        "--preview_mean",
        type=float,
        nargs=3,
        default=None,
        help="Mean for unnormalizing tensor previews (3 floats).",
    )
    parser.add_argument(
        "--preview_std",
        type=float,
        nargs=3,
        default=None,
        help="Std for unnormalizing tensor previews (3 floats).",
    )
    args = parser.parse_args()

    map_location = "cpu"
    if torch.cuda.is_available():
        map_location = torch.device("cuda:0")
    model = VehiInfoRet.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=map_location,
    )

    label_map = None
    if args.label_map_json is not None:
        label_map = _load_label_map(args.label_map_json)

    preview_index = {}
    if label_map is None:
        _, preview_index = _build_label_index(args.data_path / "labels")

    items = _collect_images_and_labels(args.data_path, label_map)
    if args.max_samples and args.max_samples > 0:
        items = items[: args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.features_cache.exists():
        cached = torch.load(args.features_cache, map_location="cpu")
        features = cached["features"]
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        labels = cached.get("labels", [])
        cached_stems = cached.get("stems", [])
        if cached_stems:
            items = [(Path(stem), "unknown") for stem in cached_stems]
        img_paths = cached.get("img_paths", [])
    else:
        features, labels, img_paths = _extract_features(
            model,
            items,
            args.batch_size,
            device,
            preview_from_tensor=args.preview_from_tensor,
            preview_mean=args.preview_mean,
            preview_std=args.preview_std,
        )
        stems = [p.stem for p, _ in items]
        torch.save(
            {"features": torch.from_numpy(features), "labels": labels, "stems": stems},
            args.features_cache,
        )
    coords = _tsne_3d(features, seed=args.seed)

    preview_from_tensor = args.preview_from_tensor
    if (
        not preview_from_tensor
        and args.preview_images_dir is None
        and args.preview_base_dir is None
        and not preview_index
    ):
        preview_from_tensor = True

    if preview_from_tensor:
        preview_paths = img_paths
    else:
        preview_paths = []
        if args.preview_images_dir is not None:
            image_index = _build_image_index(args.preview_images_dir)
            for path, _ in tqdm(items, desc="Matching previews"):
                preview = image_index.get(path.stem, "")
                if not preview and preview_index:
                    rel = preview_index.get(path.stem, "")
                    if rel:
                        preview = image_index.get(Path(rel).stem, "")
                preview_paths.append(preview)
        elif preview_index:
            base_dir = args.preview_base_dir or args.data_path
            for path, _ in tqdm(items, desc="Matching previews"):
                rel = preview_index.get(path.stem, "")
                if rel:
                    candidate = (base_dir / rel).resolve()
                    if candidate.exists():
                        preview_paths.append(str(candidate))
                    else:
                        preview_paths.append("")
                else:
                    preview_paths.append("")
        else:
            preview_paths = [""] * len(items)

    img_names = [Path(p).name for p, _ in items]
    _write_html(
        coords,
        labels,
        preview_paths,
        img_names,
        args.output_html,
        args.use_thumbnails,
    )
    print(f"Wrote: {args.output_html}")


if __name__ == "__main__":
    main()
