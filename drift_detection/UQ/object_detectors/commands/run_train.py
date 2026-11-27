from pathlib import Path

from commands.train.common import resolve_device, set_seed, validate_training_config
from commands.train.registry import normalize_train_model_type, resolve_train_runner


def run_train(config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed")
    epochs, lr, weight_decay = validate_training_config(config)
    set_seed(seed)

    device = resolve_device(config.get("model", {}).get("device", "cuda"))
    print(f"[train] device={device}")
    if str(device) == "cpu":
        print("[train][warn] CUDA unavailable -> training on CPU (very slow).")

    model_type = normalize_train_model_type(config.get("model", {}).get("type", "yolov5"))
    runner = resolve_train_runner(model_type)
    runner(
        config=config,
        run_dir=run_dir,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )


__all__ = ["run_train"]
