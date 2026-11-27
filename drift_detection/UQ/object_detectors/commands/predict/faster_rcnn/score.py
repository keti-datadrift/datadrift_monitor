from commands.predict.faster_rcnn.deterministic import run_deterministic_uncertainties_csv


def run_score_csv(config, run_dir):
    return run_deterministic_uncertainties_csv(config, run_dir, uncertainties=["score"])


__all__ = ["run_score_csv"]
