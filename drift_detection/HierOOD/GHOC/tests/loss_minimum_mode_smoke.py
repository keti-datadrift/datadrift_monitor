from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss.HiMSmin import HiMS_min_loss
from loss.WeiHiMS import HiMS_min_wei_loss


def main() -> None:
    torch.manual_seed(42)
    features = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    path_labels = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3],
        ],
        dtype=torch.long,
    )

    for loss_fn in (HiMS_min_loss, HiMS_min_wei_loss):
        batch_loss = loss_fn(features, path_labels, minimum_mode="batch")
        sample_loss = loss_fn(features, path_labels, minimum_mode="sample")
        assert batch_loss.ndim == 0 and torch.isfinite(batch_loss)
        assert sample_loss.ndim == 0 and torch.isfinite(sample_loss)

        try:
            loss_fn(features, path_labels, minimum_mode="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid minimum_mode must raise ValueError")

    print("Loss minimum-mode smoke checks passed")


if __name__ == "__main__":
    main()
