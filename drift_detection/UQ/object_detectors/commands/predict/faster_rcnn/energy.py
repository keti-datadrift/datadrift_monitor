from commands.predict.faster_rcnn.deterministic import run_deterministic_uncertainties_csv


def run_energy_csv(config, run_dir):
    return run_deterministic_uncertainties_csv(config, run_dir, uncertainties=["energy"])


__all__ = ["run_energy_csv"]
