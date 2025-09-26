import argparse
import yaml
from lightning import seed_everything
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg, DEFAULT_CFG

from trainers.trainer import YOLOTrainer
from trainers.drift_adaptation_trainer import DriftAdaptTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training / Drift Adaptation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()

if __name__ == "__main__":
    args_cli = parse_args()
    with open(args_cli.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    main_cfg = yaml_cfg["main"]
    trainer_cfg = yaml_cfg["trainer_args"]
    trainer_custom_cfg = yaml_cfg["trainer_custom"]
    
    mode = main_cfg["mode"]
    seed = main_cfg["seed"]
    seed_everything(seed, workers=True)

    base_cfg = get_cfg(DEFAULT_CFG)
    config = get_cfg(base_cfg, overrides=trainer_cfg)
    if mode == "pretrain":
        print("▶ Mode: YOLO Pretraining")
        data = check_det_dataset(config.data)
        trainer = YOLOTrainer(args=config, data=data, custom_cfg=trainer_custom_cfg)
        trainer.train()
    elif mode == "drift_adaptation":
        print("▶ Mode: Drift Adaptation")
        trainer = DriftAdaptTrainer(args=config, custom_cfg=trainer_custom_cfg, main_cfg=main_cfg)
        trainer.drift_adaptation(method=main_cfg["method"])
    else:
        raise ValueError(f"Invalid mode in config: {mode}. Choose 'pretrain' or 'drift_adaptation'.")
