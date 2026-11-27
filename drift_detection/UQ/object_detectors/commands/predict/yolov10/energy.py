import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.features import yolov10_raw_logits_for_item
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import build_detector


def run_energy_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "energy"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    output_csv = run_dir / "energy.csv"
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "energy"]
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=["detector_inference_sec"], device=device)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolov10_forward(detector, infer_batch, timing=timing)
            batch_items = 0
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                logits = yolov10_raw_logits_for_item(forward, item, device)
                energy = float((-torch.logsumexp(logits, dim=-1)).detach().cpu().item())
                row = dict(item["base_row"])
                row["energy"] = energy
                writer.writerow(row)
                batch_items += 1
            timing.record(len(image_list), batch_items, {"detector_inference_sec": forward.detector_inference_sec})
            del infer_batch, forward
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_energy_csv"]
