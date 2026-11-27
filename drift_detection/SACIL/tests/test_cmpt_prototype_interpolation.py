from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMPT_SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(CMPT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CMPT_SOURCE_ROOT))

from sacil.cmpt import (  # noqa: E402
    CMPTExperimentSettings,
    apply_low_rank_moment_transport,
    apply_polynomial_kernel_moment_transport,
    adaptive_alpha_from_uncertainties,
    build_old_class_adaptive_means,
    build_old_class_interpolated_means,
    class_geometric_oracle_alphas,
    empirical_bayes_shrink_alphas,
    fit_multiview_persistent_quadrature_weights,
    fit_low_rank_moment_transport,
    fit_low_rank_moment_transport_grid,
    fit_moment_calibration_weights,
    fit_polynomial_kernel_moment_transport,
    fit_persistent_quadrature_weights,
    herding_prefix_extrapolated_means,
    project_probability_simplex,
    transport_first_second_moments_affine,
    weighted_class_prototypes,
)
from sacil.methods.prototype_transport import (  # noqa: E402
    affine_class_residual_transport,
    apply_affine_mapping,
    classwise_translation_transport,
    local_neighbor_affine_transport,
    population_weighted_affine_transport,
)
from sacil.cmpt.drift_grouping import (  # noqa: E402
    direction_similarity_correlation,
)


def test_interpolation_endpoints_and_current_rows_are_exact() -> None:
    baseline = F.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
        ),
        dim=1,
    )
    transported = F.normalize(
        torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        dim=1,
    )

    alpha_zero = build_old_class_interpolated_means(
        baseline, transported, old_class_count=2, alpha=0.0
    )
    alpha_one = build_old_class_interpolated_means(
        baseline, transported, old_class_count=2, alpha=1.0
    )
    alpha_half = build_old_class_interpolated_means(
        baseline, transported, old_class_count=2, alpha=0.5
    )

    assert torch.equal(alpha_zero, baseline)
    assert torch.equal(alpha_one[:2], transported[:2])
    assert torch.equal(alpha_one[2:], baseline[2:])
    assert torch.equal(alpha_half[2:], baseline[2:])
    expected = F.normalize(baseline[:2] + transported[:2], dim=1)
    assert torch.allclose(alpha_half[:2], expected)


def test_interpolation_rejects_invalid_alpha() -> None:
    means = torch.eye(2)
    with pytest.raises(ValueError, match="alpha"):
        build_old_class_interpolated_means(
            means, means, old_class_count=1, alpha=1.01
        )


def test_settings_require_sweep_endpoints() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "prototype_interpolation_alphas": [0.25, 0.5, 0.75],
        },
    }
    with pytest.raises(ValueError, match="alpha=0 and alpha=1"):
        CMPTExperimentSettings.from_config(config, PROJECT_ROOT)


def test_uncertainty_ratio_weights_the_more_reliable_estimator() -> None:
    memory = torch.tensor([4.0, 1.0, 1.0])
    transport = torch.tensor([1.0, 4.0, 1.0])
    alpha = adaptive_alpha_from_uncertainties(memory, transport)
    assert torch.allclose(alpha, torch.tensor([0.8, 0.2, 0.5]))


def test_class_adaptive_interpolation_keeps_new_rows() -> None:
    baseline = torch.eye(4)
    transported = torch.flip(torch.eye(4), dims=(1,))
    result = build_old_class_adaptive_means(
        baseline,
        transported,
        old_class_count=2,
        old_class_alphas=torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(result[0], baseline[0])
    assert torch.equal(result[1], transported[1])
    assert torch.equal(result[2:], baseline[2:])


def test_empirical_bayes_shrinkage_respects_class_reliability() -> None:
    raw = torch.tensor([0.2, 0.8])
    jackknife = torch.tensor([1.0e-5, 0.2])
    shrunken, credibility, between = empirical_bayes_shrink_alphas(
        raw, jackknife, session_alpha=0.5
    )
    assert between > 0.0
    assert credibility[0] > credibility[1]
    assert abs(float(shrunken[1]) - 0.5) < abs(float(raw[1]) - 0.5)


def test_settings_parse_adaptive_alpha() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {"adaptive_alpha": {"enabled": True, "folds": 4}},
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.adaptive_alpha_enabled
    assert settings.adaptive_alpha_folds == 4


def test_class_geometric_oracle_selects_closest_segment_point() -> None:
    baseline = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    transported = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    target = torch.stack(
        [
            F.normalize(torch.tensor([1.0, 1.0]), dim=0),
            torch.tensor([0.0, 1.0]),
        ]
    )
    alphas, diagnostics = class_geometric_oracle_alphas(
        baseline,
        transported,
        target,
        alpha_grid=[0.0, 0.5, 1.0],
    )
    assert torch.equal(alphas, torch.tensor([0.5, 1.0]))
    assert diagnostics["oracle_mean_cosine_distance"] == pytest.approx(0.0)


def test_settings_parse_accuracy_oracle_grid() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "prototype_interpolation_alphas": [0.0, 0.5, 1.0],
            "oracle_diagnostics": {
                "accuracy": True,
                "alpha_grid": [0.0, 0.5, 1.0],
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.accuracy_oracle_enabled
    assert settings.oracle_alpha_grid == (0.0, 0.5, 1.0)


def test_settings_parse_full_mean_oracle_without_alpha_grid() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {"oracle_diagnostics": {"full_mean": True}},
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.full_mean_oracle_enabled
    assert settings.oracle_alpha_grid == ()


def test_herding_prefix_extrapolation_matches_richardson_formula() -> None:
    class_zero = torch.tensor(
        [[1.0, float(index)] for index in range(4)]
    )
    class_one = torch.tensor(
        [[float(index), 1.0] for index in range(4)]
    )
    features = torch.cat([class_zero, class_one], dim=0)
    targets = torch.tensor([0] * 4 + [1] * 4)
    ordinary, extrapolated = herding_prefix_extrapolated_means(
        features,
        targets,
        2,
        full_prefix=4,
        reference_prefix=2,
    )
    expected_ordinary = F.normalize(
        torch.stack([class_zero.mean(0), class_one.mean(0)]), dim=1
    )
    expected_extrapolated = F.normalize(
        torch.stack(
            [
                2.0 * class_zero.mean(0) - class_zero[:2].mean(0),
                2.0 * class_one.mean(0) - class_one[:2].mean(0),
            ]
        ),
        dim=1,
    )
    assert torch.allclose(ordinary, expected_ordinary)
    assert torch.allclose(extrapolated, expected_extrapolated)


def test_settings_parse_herding_extrapolation() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
            "expected_exemplars_per_class": 20,
        },
        "cmpt": {
            "herding_extrapolation": {
                "enabled": True,
                "full_prefix": 20,
                "reference_prefix": 10,
            }
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.herding_extrapolation_enabled
    assert settings.herding_full_prefix == 20
    assert settings.herding_reference_prefix == 10


def test_simplex_projection_and_quadrature_reconstruct_population_mean() -> None:
    projected = project_probability_simplex(
        torch.tensor([-0.4, 0.7, 1.3])
    )
    assert torch.all(projected >= 0.0)
    assert float(projected.sum()) == pytest.approx(1.0)

    exemplars = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    population = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    weights, diagnostics = fit_persistent_quadrature_weights(
        exemplars,
        population,
        uniform_ridge=1.0e-6,
        max_iterations=2000,
    )
    assert float(weights.sum()) == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["weighted_target_cosine_distance"] < diagnostics[
        "uniform_target_cosine_distance"
    ]


def test_persistent_weights_apply_classwise_in_current_space() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
    prototypes = weighted_class_prototypes(features, targets, weights)
    expected = F.normalize(
        torch.tensor([[0.75, 0.25], [-0.25, -0.75]]), dim=1
    )
    assert torch.allclose(prototypes, expected)


def test_multiview_quadrature_uses_one_weight_vector_for_all_views() -> None:
    exemplar_views = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    population_views = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
        ]
    )
    weights, diagnostics = fit_multiview_persistent_quadrature_weights(
        exemplar_views,
        population_views,
        uniform_ridge=1.0e-5,
        max_iterations=2000,
    )
    assert weights.shape == (3,)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["view_count"] == 2
    assert diagnostics["weighted_target_cosine_distance"] < diagnostics[
        "uniform_target_cosine_distance"
    ]


def test_settings_parse_persistent_quadrature() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "persistent_quadrature": {
                "enabled": True,
                "uniform_ridge": 0.002,
                "max_iterations": 321,
                "multiview": True,
                "previous_model_view": True,
            }
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.persistent_quadrature_enabled
    assert settings.persistent_quadrature_ridge == pytest.approx(0.002)
    assert settings.persistent_quadrature_iterations == 321
    assert settings.persistent_quadrature_multiview
    assert settings.persistent_quadrature_previous_model_view


def test_settings_parse_canonical_reference_transport() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "affine_ridge": 0.01,
            "canonical_reference": {
                "enabled": True,
                "affine_ridge": 0.02,
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.canonical_reference_enabled
    assert settings.canonical_reference_ridge == pytest.approx(0.02)


def test_settings_parse_moment_calibrated_affine() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "moment_calibrated_affine": {
                "enabled": True,
                "uniform_ridge": 0.002,
                "max_iterations": 321,
                "affine_ridge": 0.03,
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.moment_calibrated_affine_enabled
    assert settings.moment_calibrated_affine_uniform_ridge == pytest.approx(
        0.002
    )
    assert settings.moment_calibrated_affine_iterations == 321
    assert settings.moment_calibrated_affine_ridge == pytest.approx(0.03)


def test_moment_calibration_improves_requested_population_moments() -> None:
    features = torch.tensor(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float32
    )
    targets = torch.zeros(3, dtype=torch.long)
    mean = torch.tensor([[0.2, 0.0]])
    covariance = torch.tensor([[[0.85, 0.0], [0.0, 0.0]]])
    second = covariance + torch.einsum("cd,ce->cde", mean, mean)
    uniform = torch.full((3,), 1.0 / 3.0)

    mean_weights, mean_stats = fit_moment_calibration_weights(
        features,
        targets,
        mean,
        second,
        mode="mean",
        uniform_ridge=1.0e-4,
    )
    second_weights, second_stats = fit_moment_calibration_weights(
        features,
        targets,
        mean,
        second,
        mode="second",
        uniform_ridge=1.0e-4,
    )
    combined_weights, combined_stats = fit_moment_calibration_weights(
        features,
        targets,
        mean,
        second,
        mode="combined",
        uniform_ridge=1.0e-4,
    )

    assert torch.allclose(mean_weights.sum(), torch.tensor(1.0))
    assert torch.allclose(second_weights.sum(), torch.tensor(1.0))
    assert torch.allclose(combined_weights.sum(), torch.tensor(1.0))
    assert not torch.allclose(mean_weights, uniform)
    assert mean_stats["mean_weighted_mean_relative_error"] < mean_stats[
        "mean_uniform_mean_relative_error"
    ]
    assert second_stats["mean_weighted_second_relative_error"] < second_stats[
        "mean_uniform_second_relative_error"
    ]
    assert combined_stats["mean_weighted_mean_relative_error"] < (
        combined_stats["mean_uniform_mean_relative_error"]
    )
    assert combined_stats["mean_weighted_second_relative_error"] < (
        combined_stats["mean_uniform_second_relative_error"]
    )


def test_affine_population_moment_transport_matches_samples() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(1000, 3, generator=generator)
    mapping = torch.tensor(
        [
            [1.2, 0.1, 0.0],
            [-0.2, 0.8, 0.1],
            [0.0, 0.3, 1.1],
            [0.2, -0.1, 0.05],
        ]
    )
    target = source @ mapping[:-1] + mapping[-1]
    means = source.mean(dim=0, keepdim=True)
    seconds = torch.einsum("nd,ne->de", source, source).unsqueeze(0) / len(
        source
    )
    transported_mean, transported_second = (
        transport_first_second_moments_affine(means, seconds, mapping)
    )
    expected_mean = target.mean(dim=0, keepdim=True)
    expected_second = (
        torch.einsum("nd,ne->de", target, target).unsqueeze(0) / len(target)
    )
    assert torch.allclose(transported_mean, expected_mean, atol=1.0e-5)
    assert torch.allclose(transported_second, expected_second, atol=1.0e-5)


def test_population_weighted_affine_changes_fit_measure() -> None:
    prototypes = torch.tensor([[1.0, 0.0]])
    old = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    current = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]]
    )
    uniform, _, _ = population_weighted_affine_transport(
        prototypes,
        old,
        current,
        torch.ones(4),
        ridge=0.1,
    )
    weighted, _, _ = population_weighted_affine_transport(
        prototypes,
        old,
        current,
        torch.tensor([10.0, 1.0, 1.0, 1.0]),
        ridge=0.1,
    )
    assert not torch.allclose(uniform, weighted)


def test_population_mass_transport_requires_quadrature() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "population_mass_transport": {"enabled": True},
        },
    }
    with pytest.raises(ValueError, match="requires persistent quadrature"):
        CMPTExperimentSettings.from_config(config, PROJECT_ROOT)


def test_classwise_translation_uses_matched_class_mean_drifts() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    current = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])

    transported, translations = classwise_translation_transport(
        prototypes, old, current, targets
    )

    assert torch.allclose(
        translations,
        torch.tensor([[-1.0, 1.0], [-1.0, -1.0]]),
    )
    assert torch.allclose(transported, current[::2])


def test_affine_class_residual_is_zero_for_exact_global_map() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    current = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    mapping = torch.tensor(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    combined, residuals = affine_class_residual_transport(
        prototypes, mapping, old, current, targets
    )
    global_only = apply_affine_mapping(prototypes, mapping)

    assert torch.allclose(residuals, torch.zeros_like(residuals))
    assert torch.allclose(combined, global_only)


def test_affine_class_residual_adds_observed_class_offset() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = prototypes.repeat_interleave(2, dim=0)
    current = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    identity_affine = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    combined, residuals = affine_class_residual_transport(
        prototypes, identity_affine, old, current, targets
    )
    expected_current_means = F.normalize(current[::2], dim=1)

    assert bool((residuals.norm(dim=1) > 0.0).all())
    assert torch.allclose(combined, expected_current_means)


def test_settings_parse_component_ablation() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "component_ablation": {"enabled": True},
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.component_ablation_enabled


def test_drift_similarity_correlation_detects_matched_geometry() -> None:
    prototypes = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        dim=1,
    )
    result = direction_similarity_correlation(
        prototypes,
        prototypes,
        permutations=100,
        seed=7,
    )
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["permutation_p_positive"] <= 0.05


def test_local_neighbor_affine_uses_center_and_nearest_classes() -> None:
    prototypes = F.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]]
        ),
        dim=1,
    )
    old = prototypes.repeat_interleave(2, dim=0)
    current = old.clone()
    targets = torch.arange(4).repeat_interleave(2)

    transported, neighborhoods, residuals = local_neighbor_affine_transport(
        prototypes,
        old,
        current,
        targets,
        classes_per_neighborhood=2,
        ridge=1.0e-6,
    )

    assert set(neighborhoods[0].tolist()) == {0, 1}
    assert set(neighborhoods[2].tolist()) == {2, 3}
    assert torch.allclose(transported, prototypes, atol=2.0e-3)
    assert bool((residuals < 1.0e-5).all())


def test_settings_parse_neighbor_affine() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "neighbor_affine": {
                "enabled": True,
                "classes_per_neighborhood": 5,
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.neighbor_affine_enabled
    assert settings.neighbor_affine_classes == 5


def test_moment_transport_uses_second_moment_for_quadratic_drift() -> None:
    generator = torch.Generator().manual_seed(19)
    support = torch.randn(800, 3, generator=generator)
    current = support.clone()
    current[:, 0] += 0.8 * support[:, 1].square()
    mapping = fit_low_rank_moment_transport(
        support,
        current,
        rank=3,
        affine_ridge=1.0e-5,
        quadratic_ridge=1.0e-5,
    )

    populations = torch.stack(
        [
            torch.randn(3000, 3, generator=generator)
            + torch.tensor([-1.0, 0.2, 0.0]),
            1.7 * torch.randn(3000, 3, generator=generator)
            + torch.tensor([1.0, -0.3, 0.4]),
        ]
    )
    means = populations.mean(dim=1)
    seconds = torch.einsum(
        "cnd,cne->cde", populations, populations
    ) / populations.shape[1]
    true_current = populations.clone()
    true_current[:, :, 0] += 0.8 * populations[:, :, 1].square()
    true_means = true_current.mean(dim=1)

    transported, _, _ = apply_low_rank_moment_transport(
        means, seconds, mapping
    )
    affine = mapping.affine_mapping
    affine_only = means @ affine[:-1] + affine[-1]
    moment_error = (transported - true_means).norm(dim=1).mean()
    affine_error = (affine_only - true_means).norm(dim=1).mean()

    assert mapping.quadratic_fit_residual < 0.01 * mapping.affine_fit_residual
    assert moment_error < 0.1 * affine_error


def test_polynomial_kernel_transport_uses_full_space_second_moment() -> None:
    generator = torch.Generator().manual_seed(23)
    support = torch.randn(500, 3, generator=generator)
    current = support.clone()
    current[:, 0] += 0.7 * support[:, 1].square()
    current[:, 2] += 0.4 * support[:, 0] * support[:, 2]
    mapping = fit_polynomial_kernel_moment_transport(
        support,
        current,
        affine_ridge=1.0e-5,
        quadratic_ridge=1.0e-3,
    )

    populations = torch.stack(
        [
            torch.randn(2500, 3, generator=generator)
            + torch.tensor([-0.7, 0.4, 0.1]),
            1.4 * torch.randn(2500, 3, generator=generator)
            + torch.tensor([0.8, -0.2, 0.5]),
        ]
    )
    means = populations.mean(dim=1)
    seconds = torch.einsum(
        "cnd,cne->cde", populations, populations
    ) / populations.shape[1]
    true_current = populations.clone()
    true_current[:, :, 0] += 0.7 * populations[:, :, 1].square()
    true_current[:, :, 2] += (
        0.4 * populations[:, :, 0] * populations[:, :, 2]
    )
    true_means = true_current.mean(dim=1)

    transported, _, _ = apply_polynomial_kernel_moment_transport(
        means, seconds, mapping
    )
    affine = mapping.affine_mapping
    affine_only = means @ affine[:-1] + affine[-1]
    kernel_error = (transported - true_means).norm(dim=1).mean()
    affine_error = (affine_only - true_means).norm(dim=1).mean()

    assert mapping.support_count == support.shape[0]
    assert mapping.quadratic_fit_residual < 0.02 * mapping.affine_fit_residual
    assert kernel_error < 0.1 * affine_error


def test_moment_grid_shares_fit_and_scales_quadratic_correction() -> None:
    generator = torch.Generator().manual_seed(31)
    old = torch.randn(160, 5, generator=generator)
    current = old.clone()
    current[:, 0] += 0.3 * old[:, 1].square()
    mappings = fit_low_rank_moment_transport_grid(
        old,
        current,
        ranks=(2, 4),
        quadratic_ridges=(0.1, 1.0),
        affine_ridge=0.01,
    )
    assert set(mappings) == {(2, 0.1), (2, 1.0), (4, 0.1), (4, 1.0)}

    individual = fit_low_rank_moment_transport(
        old,
        current,
        rank=4,
        affine_ridge=0.01,
        quadratic_ridge=0.1,
    )
    shared = mappings[(4, 0.1)]
    assert torch.allclose(
        shared.quadratic_mapping,
        individual.quadratic_mapping,
        atol=1.0e-6,
    )

    populations = torch.randn(3, 80, 5, generator=generator)
    means = populations.mean(dim=1)
    seconds = torch.einsum(
        "cnd,cne->cde", populations, populations
    ) / populations.shape[1]
    affine_only, _, _ = apply_low_rank_moment_transport(
        means,
        seconds,
        shared,
        correction_scale=0.0,
    )
    affine = shared.affine_mapping
    expected = means @ affine[:-1] + affine[-1]
    assert torch.allclose(affine_only, expected, atol=1.0e-6)


def test_settings_parse_moment_grid() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "moment_transport": {
                "enabled": True,
                "grid": {
                    "enabled": True,
                    "ranks": [2, 6],
                    "quadratic_ridges": [0.1, 10.0],
                    "correction_scales": [0.25, 1.0],
                    "validation_folds": 4,
                    "validation_fold": 1,
                },
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.moment_grid_enabled
    assert settings.moment_grid_ranks == (2, 6)
    assert settings.moment_grid_quadratic_ridges == (0.1, 10.0)
    assert settings.moment_grid_correction_scales == (0.25, 1.0)
    assert settings.moment_grid_validation_folds == 4
    assert settings.moment_grid_validation_fold == 1


def test_settings_parse_moment_transport() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "moment_transport": {
                "enabled": True,
                "mode": "polynomial_kernel",
                "rank": 4,
                "affine_ridge": 0.02,
                "quadratic_ridge": 0.5,
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.moment_transport_enabled
    assert settings.moment_transport_mode == "polynomial_kernel"
    assert settings.moment_transport_rank == 4
    assert settings.moment_transport_affine_ridge == pytest.approx(0.02)
    assert settings.moment_transport_quadratic_ridge == pytest.approx(0.5)
