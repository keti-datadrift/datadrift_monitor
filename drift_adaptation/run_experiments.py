import subprocess
import yaml
from itertools import product
from copy import deepcopy

base_config_path = "./configs/drift_adapt.yaml"
temp_config_path = "temp.yaml"

seeds = [42, 43, 44, 45, 46]
methods = ["no_adapt", "naive1", "naive2", "gmm"]

with open(base_config_path, "r") as f:
    base_cfg = yaml.safe_load(f)

for seed, method in product(seeds, methods):
    cfg = deepcopy(base_cfg)
    cfg["main"]["seed"] = seed
    cfg["main"]["method"] = method
    
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"\n[RUN] seed={seed}, method={method}")
    subprocess.run(["python", "main.py", "--config", temp_config_path])
