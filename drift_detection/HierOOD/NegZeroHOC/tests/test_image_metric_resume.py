from __future__ import annotations

import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from negzerohoc.checkpointing import (
    load_idea3_checkpoint,
    save_idea3_checkpoint,
)
from scripts.train_idea3_joint_vision_lora import make_grad_scaler
from scripts.train_image_metric_vision_lora import (
    capture_rng_state,
    load_config,
    make_training_state,
    next_epoch_from_training_state,
    restore_rng_state,
    restore_training_components,
    resume_signature,
)


def training_args():
    return SimpleNamespace(
        experiment_name="resume-test",
        dataset="toy",
        datadir="toy-data",
        hierarchy="toy-hierarchy.json",
        id_split="toy-id.csv",
        clip_model="toy-clip",
        vision_lora={"rank": 2},
        seed=7,
        deterministic=True,
        num_workers=0,
        augmentation={"scale": [0.8, 1.0]},
        epochs=3,
        classes_per_batch=2,
        examples_per_class=2,
        eval_batch_size=4,
        lora_lr=1e-3,
        proxy_lr=2e-3,
        weight_decay=1e-4,
        precision="fp32",
        gradient_checkpointing=False,
        gradient_clip_norm=1.0,
        supcon_temperature=0.1,
        proxy_temperature=0.07,
        proxy_margin=0.05,
        triplet_base_margin=0.1,
        triplet_hierarchy_margin=0.1,
        lambda_supcon=1.0,
        lambda_triplet=0.5,
        lambda_proxy=1.0,
        lambda_retention=0.5,
        validation_every_n_epochs=1,
        reference_only_training=True,
        reference_fraction=0.8,
    )


def make_components():
    model = torch.nn.Linear(3, 2)
    proxy = torch.nn.Parameter(torch.randn(2))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-3},
            {"params": [proxy], "lr": 2e-3},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3
    )
    scaler = make_grad_scaler(False)
    loader = SimpleNamespace(
        generator=torch.Generator().manual_seed(101),
        batch_sampler=SimpleNamespace(),
    )
    return model, proxy, optimizer, scheduler, scaler, loader


def train_epoch(model, proxy, optimizer, scheduler, loader):
    optimizer.zero_grad(set_to_none=True)
    python_noise = random.random()
    numpy_noise = float(np.random.rand())
    loader_noise = float(
        torch.rand((), generator=loader.generator)
    )
    inputs = torch.rand(4, 3)
    loss = (
        model(inputs).square().mean()
        + proxy.square().mean()
        + 0.0 * (python_noise + numpy_noise + loader_noise)
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second):
            assert_nested_equal(first_item, second_item)
    else:
        assert first == second


def test_image_metric_resume_is_exact_at_epoch_boundary():
    args = training_args()
    classes = ["a", "b"]
    with mock.patch("torch.cuda.is_available", return_value=False):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        (
            model,
            proxy,
            optimizer,
            scheduler,
            scaler,
            loader,
        ) = make_components()
        history = [{"epoch": 1, "loss": train_epoch(
            model, proxy, optimizer, scheduler, loader
        )}]
        model_at_boundary = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        proxy_at_boundary = proxy.detach().cpu().clone()
        training_state = make_training_state(
            args,
            classes=classes,
            epoch=1,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=loader,
            device="cpu",
            history=history,
            best_epoch=1,
            best_bacc=0.5,
            best_lora_state=model_at_boundary,
            best_proxy_state=proxy_at_boundary,
            training_loop_complete=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "last.pt"
            )
            save_idea3_checkpoint(
                checkpoint_path,
                stage="image_metric_vision_lora",
                dataset=args.dataset,
                clip_model=args.clip_model,
                hierarchy=args.hierarchy,
                id_split=args.id_split,
                prompt_config={},
                vision_lora_config=args.vision_lora,
                vision_lora_state_dict=model_at_boundary,
                training_state=training_state,
                extra_payload={
                    "metric_proxies": proxy_at_boundary,
                    "metric_proxy_classes": classes,
                },
            )

            uninterrupted_losses = []
            for _ in range(2, 4):
                uninterrupted_losses.append(
                    train_epoch(
                        model,
                        proxy,
                        optimizer,
                        scheduler,
                        loader,
                    )
                )
            uninterrupted_model = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            uninterrupted_proxy = proxy.detach().cpu().clone()
            uninterrupted_optimizer = optimizer.state_dict()
            uninterrupted_scheduler = scheduler.state_dict()
            uninterrupted_rng = capture_rng_state(loader, "cpu")

            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            (
                resumed_model,
                resumed_proxy,
                resumed_optimizer,
                resumed_scheduler,
                resumed_scaler,
                resumed_loader,
            ) = make_components()
            payload = load_idea3_checkpoint(checkpoint_path)
            resumed_model.load_state_dict(
                payload["vision_lora_state_dict"]
            )
            with torch.no_grad():
                resumed_proxy.copy_(payload["metric_proxies"])
            restore_training_components(
                payload["training_state"],
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                scaler=resumed_scaler,
                train_loader=resumed_loader,
                device="cpu",
            )
            resumed_losses = []
            for _ in range(2, 4):
                resumed_losses.append(
                    train_epoch(
                        resumed_model,
                        resumed_proxy,
                        resumed_optimizer,
                        resumed_scheduler,
                        resumed_loader,
                    )
                )

            assert resumed_losses == uninterrupted_losses
            assert_nested_equal(
                resumed_model.state_dict(), uninterrupted_model
            )
            assert torch.equal(resumed_proxy, uninterrupted_proxy)
            assert_nested_equal(
                resumed_optimizer.state_dict(),
                uninterrupted_optimizer,
            )
            assert_nested_equal(
                resumed_scheduler.state_dict(),
                uninterrupted_scheduler,
            )
            resumed_rng = capture_rng_state(
                resumed_loader, "cpu"
            )
            assert_nested_equal(
                resumed_rng["torch"], uninterrupted_rng["torch"]
            )
            assert_nested_equal(
                resumed_rng["train_loader_generator"],
                uninterrupted_rng["train_loader_generator"],
            )


def test_rng_round_trip_includes_python_numpy_torch_and_loader():
    loader = SimpleNamespace(
        generator=torch.Generator().manual_seed(31)
    )
    with mock.patch("torch.cuda.is_available", return_value=False):
        random.seed(11)
        np.random.seed(13)
        torch.manual_seed(17)
        state = capture_rng_state(loader, "cpu")
        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
            torch.rand(3, generator=loader.generator),
        )
        restore_rng_state(state, loader, "cpu")
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
            torch.rand(3, generator=loader.generator),
        )

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3])


def test_complete_checkpoint_skips_to_finalization():
    assert next_epoch_from_training_state(
        {"epoch": 2, "training_loop_complete": False}, 3
    ) == 3
    assert next_epoch_from_training_state(
        {"epoch": 3, "training_loop_complete": True}, 3
    ) == 4


def test_signature_changes_when_trajectory_setting_changes():
    args = training_args()
    original = resume_signature(args, ["a", "b"])
    args.lambda_proxy = 2.0
    assert resume_signature(args, ["a", "b"]) != original


def test_image_metric_resume_is_opt_in_by_default():
    args = load_config(
        "configs/09_image_metric_joint_prompts/"
        "idea8_fgvc_aircraft_b16_image_metric_lora.yaml"
    )
    assert args.resume_enabled is False
    assert args.resume_checkpoint == args.last_checkpoint
