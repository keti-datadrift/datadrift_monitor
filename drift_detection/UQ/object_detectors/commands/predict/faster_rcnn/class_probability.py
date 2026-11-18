from commands.predict.faster_rcnn.deterministic import run_deterministic_uncertainties_csv


def run_class_probability_csv(config, run_dir):
    return run_deterministic_uncertainties_csv(config, run_dir, uncertainties=["class_probability"])


__all__ = ["run_class_probability_csv"]
