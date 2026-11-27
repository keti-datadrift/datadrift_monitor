from __future__ import annotations

import copy
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sacil.anchors import compute_prototypes
from sacil.engine.checkpoint import load_checkpoint
from sacil.engine.evaluator import EvaluationResult
from sacil.engine.evaluator import evaluate as evaluate_classifier
from sacil.engine.evaluator import evaluate_nme
from sacil.engine.table1_trainer import UnifiedTable1Trainer
from sacil.features import collect_features
from sacil.memory import ExemplarMemory
from sacil.methods import normalized_cosine_classifier_logits
from .herding_extrapolation import (
    herding_prefix_extrapolated_means,
)
from .persistent_quadrature import (
    fit_multiview_persistent_quadrature_weights,
    weighted_class_prototypes,
)
from .moment_transport import (
    apply_low_rank_moment_transport,
    apply_polynomial_kernel_moment_transport,
    class_first_second_moments,
    fit_low_rank_moment_transport,
    fit_low_rank_moment_transport_grid,
    fit_polynomial_kernel_moment_transport,
)
from .moment_calibrated_affine import (
    CALIBRATION_MODES,
    expand_exemplar_weights,
    fit_moment_calibration_weights,
    transport_first_second_moments_affine,
)
from sacil.methods.prototype_transport import (
    affine_class_residual_transport,
    affine_ridge_transport,
    classwise_translation_transport,
    local_neighbor_affine_transport,
    population_weighted_affine_transport,
    rigid_procrustes_transport,
)
from sacil.provenance import build_exploration_provenance
from sacil.utils import dump_json, git_commit


_CHECKPOINT_PATTERN = re.compile(r"^session_(\d+)\.pt$")


@dataclass(frozen=True)
class CMPTExperimentSettings:
    learner: str
    checkpoint_directory: Path
    output_file: Path
    expected_checkpoint_method: str
    expected_sessions: int = 11
    expected_exemplars_per_class: int = 20
    expected_casper_enabled: bool = False
    device: str = "cuda:0"
    transport: str = "rigid_procrustes"
    affine_ridge: float = 1.0e-2
    prototype_horizontal_flip: bool = True
    support_horizontal_flip: bool = True
    query_horizontal_flip: bool = False
    center_strength: float = 0.0
    strict_parity: bool = True
    parity_tolerance: float = 1.0e-6
    cpu_threads: int = 6
    prototype_interpolation_alphas: tuple[float, ...] = ()
    adaptive_alpha_enabled: bool = False
    adaptive_alpha_folds: int = 5
    accuracy_oracle_enabled: bool = False
    class_geometric_oracle_enabled: bool = False
    full_mean_oracle_enabled: bool = False
    oracle_alpha_grid: tuple[float, ...] = ()
    component_ablation_enabled: bool = False
    neighbor_affine_enabled: bool = False
    neighbor_affine_classes: int = 5
    herding_extrapolation_enabled: bool = False
    herding_full_prefix: int = 20
    herding_reference_prefix: int = 10
    persistent_quadrature_enabled: bool = False
    persistent_quadrature_ridge: float = 1.0e-3
    persistent_quadrature_iterations: int = 1000
    persistent_quadrature_multiview: bool = False
    persistent_quadrature_previous_model_view: bool = False
    canonical_reference_enabled: bool = False
    canonical_reference_ridge: float = 1.0e-2
    population_mass_transport_enabled: bool = False
    population_mass_transport_ridge: float = 1.0e-2
    moment_calibrated_affine_enabled: bool = False
    moment_calibrated_affine_uniform_ridge: float = 1.0e-3
    moment_calibrated_affine_iterations: int = 1000
    moment_calibrated_affine_ridge: float = 1.0e-2
    moment_transport_enabled: bool = False
    moment_transport_mode: str = "low_rank_pca"
    moment_transport_rank: int = 6
    moment_transport_affine_ridge: float = 1.0e-2
    moment_transport_quadratic_ridge: float = 1.0
    moment_transport_residual_covariance_scale: float = 0.0
    moment_grid_enabled: bool = False
    moment_grid_ranks: tuple[int, ...] = ()
    moment_grid_quadratic_ridges: tuple[float, ...] = ()
    moment_grid_correction_scales: tuple[float, ...] = ()
    moment_grid_validation_folds: int = 5
    moment_grid_validation_fold: int = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        project_root: str | Path,
    ) -> "CMPTExperimentSettings":
        root = Path(project_root).expanduser().resolve()
        experiment = _required_mapping(config, "experiment")
        cmpt = _required_mapping(config, "cmpt")
        runtime = config.get("runtime", {})
        if runtime is None:
            runtime = {}
        if not isinstance(runtime, Mapping):
            raise ValueError("runtime must be a mapping")
        cpu_threads = int(runtime.get("cpu_threads", torch.get_num_threads()))
        if cpu_threads <= 0:
            raise ValueError("runtime.cpu_threads must be positive")

        def project_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        transport = str(cmpt.get("transport", "rigid_procrustes")).lower()
        if transport not in {"rigid_procrustes", "affine_ridge"}:
            raise ValueError(
                "cmpt.transport must be rigid_procrustes or affine_ridge"
            )
        affine_ridge = float(cmpt.get("affine_ridge", 1.0e-2))
        if affine_ridge <= 0.0:
            raise ValueError("cmpt.affine_ridge must be positive")
        if not bool(cmpt.get("full_introduction_prototypes", True)):
            raise ValueError(
                "CMPT requires full_introduction_prototypes=true"
            )
        if not bool(cmpt.get("replace_old_classes_only", True)):
            raise ValueError("CMPT primary evaluator replaces old classes only")
        center_strength = float(cmpt.get("center_strength", 0.0))
        if not 0.0 <= center_strength <= 1.0:
            raise ValueError("cmpt.center_strength must be in [0, 1]")
        parity_tolerance = float(cmpt.get("parity_tolerance", 1.0e-6))
        if parity_tolerance < 0.0:
            raise ValueError("cmpt.parity_tolerance must be non-negative")
        raw_alphas = cmpt.get("prototype_interpolation_alphas", ())
        if raw_alphas is None:
            raw_alphas = ()
        if isinstance(raw_alphas, (str, bytes)) or not isinstance(
            raw_alphas, Sequence
        ):
            raise ValueError(
                "cmpt.prototype_interpolation_alphas must be a sequence"
            )
        interpolation_alphas = tuple(float(value) for value in raw_alphas)
        if any(
            not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
            for alpha in interpolation_alphas
        ):
            raise ValueError(
                "cmpt.prototype_interpolation_alphas must lie in [0, 1]"
            )
        if len(set(interpolation_alphas)) != len(interpolation_alphas):
            raise ValueError(
                "cmpt.prototype_interpolation_alphas must be unique"
            )
        if interpolation_alphas and not (
            0.0 in interpolation_alphas and 1.0 in interpolation_alphas
        ):
            raise ValueError(
                "an interpolation sweep must include alpha=0 and alpha=1"
            )
        adaptive_alpha = cmpt.get("adaptive_alpha", {})
        if adaptive_alpha is None:
            adaptive_alpha = {}
        if not isinstance(adaptive_alpha, Mapping):
            raise ValueError("cmpt.adaptive_alpha must be a mapping")
        adaptive_alpha_folds = int(adaptive_alpha.get("folds", 5))
        if adaptive_alpha_folds < 2:
            raise ValueError("cmpt.adaptive_alpha.folds must be at least 2")
        oracle = cmpt.get("oracle_diagnostics", {})
        if oracle is None:
            oracle = {}
        if not isinstance(oracle, Mapping):
            raise ValueError("cmpt.oracle_diagnostics must be a mapping")
        raw_oracle_grid = oracle.get(
            "alpha_grid", interpolation_alphas
        )
        if raw_oracle_grid is None:
            raw_oracle_grid = ()
        if isinstance(raw_oracle_grid, (str, bytes)) or not isinstance(
            raw_oracle_grid, Sequence
        ):
            raise ValueError(
                "cmpt.oracle_diagnostics.alpha_grid must be a sequence"
            )
        oracle_alpha_grid = tuple(float(value) for value in raw_oracle_grid)
        if any(
            not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
            for alpha in oracle_alpha_grid
        ):
            raise ValueError("oracle alpha grid values must lie in [0, 1]")
        if len(set(oracle_alpha_grid)) != len(oracle_alpha_grid):
            raise ValueError("oracle alpha grid values must be unique")
        accuracy_oracle_enabled = bool(oracle.get("accuracy", False))
        class_geometric_oracle_enabled = bool(
            oracle.get("class_geometric", False)
        )
        full_mean_oracle_enabled = bool(oracle.get("full_mean", False))
        if (accuracy_oracle_enabled or class_geometric_oracle_enabled) and not (
            oracle_alpha_grid
            and 0.0 in oracle_alpha_grid
            and 1.0 in oracle_alpha_grid
        ):
            raise ValueError(
                "oracle diagnostics require a grid containing 0 and 1"
            )
        if accuracy_oracle_enabled and not interpolation_alphas:
            raise ValueError(
                "accuracy oracle requires prototype_interpolation_alphas"
            )
        if accuracy_oracle_enabled and set(oracle_alpha_grid) != set(
            interpolation_alphas
        ):
            raise ValueError(
                "accuracy oracle grid must equal the interpolation grid"
            )
        component_ablation = cmpt.get("component_ablation", {})
        if component_ablation is None:
            component_ablation = {}
        if not isinstance(component_ablation, Mapping):
            raise ValueError("cmpt.component_ablation must be a mapping")
        component_ablation_enabled = bool(
            component_ablation.get("enabled", False)
        )
        if component_ablation_enabled and transport != "affine_ridge":
            raise ValueError(
                "CMPT component ablation requires affine_ridge transport"
            )
        neighbor_affine = cmpt.get("neighbor_affine", {})
        if neighbor_affine is None:
            neighbor_affine = {}
        if not isinstance(neighbor_affine, Mapping):
            raise ValueError("cmpt.neighbor_affine must be a mapping")
        neighbor_affine_enabled = bool(neighbor_affine.get("enabled", False))
        neighbor_affine_classes = int(
            neighbor_affine.get("classes_per_neighborhood", 5)
        )
        if neighbor_affine_classes < 2:
            raise ValueError(
                "cmpt.neighbor_affine.classes_per_neighborhood must be at least 2"
            )
        if neighbor_affine_enabled and transport != "affine_ridge":
            raise ValueError("neighbor affine comparison requires affine_ridge")
        herding_extrapolation = cmpt.get("herding_extrapolation", {})
        if herding_extrapolation is None:
            herding_extrapolation = {}
        if not isinstance(herding_extrapolation, Mapping):
            raise ValueError("cmpt.herding_extrapolation must be a mapping")
        herding_extrapolation_enabled = bool(
            herding_extrapolation.get("enabled", False)
        )
        herding_full_prefix = int(
            herding_extrapolation.get(
                "full_prefix",
                experiment.get("expected_exemplars_per_class", 20),
            )
        )
        herding_reference_prefix = int(
            herding_extrapolation.get(
                "reference_prefix", herding_full_prefix // 2
            )
        )
        if not 0 < herding_reference_prefix < herding_full_prefix:
            raise ValueError(
                "herding reference_prefix must lie between 0 and full_prefix"
            )
        expected_exemplars = int(
            experiment.get("expected_exemplars_per_class", 20)
        )
        if herding_extrapolation_enabled and (
            herding_full_prefix != expected_exemplars
        ):
            raise ValueError(
                "herding full_prefix must equal expected_exemplars_per_class"
            )
        persistent_quadrature = cmpt.get("persistent_quadrature", {})
        if persistent_quadrature is None:
            persistent_quadrature = {}
        if not isinstance(persistent_quadrature, Mapping):
            raise ValueError("cmpt.persistent_quadrature must be a mapping")
        persistent_quadrature_enabled = bool(
            persistent_quadrature.get("enabled", False)
        )
        persistent_quadrature_ridge = float(
            persistent_quadrature.get("uniform_ridge", 1.0e-3)
        )
        persistent_quadrature_iterations = int(
            persistent_quadrature.get("max_iterations", 1000)
        )
        persistent_quadrature_multiview = bool(
            persistent_quadrature.get("multiview", False)
        )
        persistent_quadrature_previous_model_view = bool(
            persistent_quadrature.get("previous_model_view", False)
        )
        if (
            persistent_quadrature_previous_model_view
            and not persistent_quadrature_multiview
        ):
            raise ValueError(
                "previous_model_view requires persistent quadrature multiview"
            )
        if persistent_quadrature_ridge < 0.0:
            raise ValueError(
                "persistent quadrature uniform_ridge must be non-negative"
            )
        if persistent_quadrature_iterations <= 0:
            raise ValueError(
                "persistent quadrature max_iterations must be positive"
            )
        canonical_reference = cmpt.get("canonical_reference", {})
        if canonical_reference is None:
            canonical_reference = {}
        if not isinstance(canonical_reference, Mapping):
            raise ValueError("cmpt.canonical_reference must be a mapping")
        canonical_reference_enabled = bool(
            canonical_reference.get("enabled", False)
        )
        canonical_reference_ridge = float(
            canonical_reference.get("affine_ridge", affine_ridge)
        )
        if canonical_reference_ridge <= 0.0:
            raise ValueError(
                "canonical reference affine_ridge must be positive"
            )
        population_mass_transport = cmpt.get(
            "population_mass_transport", {}
        )
        if population_mass_transport is None:
            population_mass_transport = {}
        if not isinstance(population_mass_transport, Mapping):
            raise ValueError(
                "cmpt.population_mass_transport must be a mapping"
            )
        population_mass_transport_enabled = bool(
            population_mass_transport.get("enabled", False)
        )
        population_mass_transport_ridge = float(
            population_mass_transport.get("affine_ridge", affine_ridge)
        )
        if population_mass_transport_ridge <= 0.0:
            raise ValueError(
                "population-mass transport affine_ridge must be positive"
            )
        if (
            population_mass_transport_enabled
            and not persistent_quadrature_enabled
        ):
            raise ValueError(
                "population-mass transport requires persistent quadrature"
            )
        moment_calibrated_affine = cmpt.get(
            "moment_calibrated_affine", {}
        )
        if moment_calibrated_affine is None:
            moment_calibrated_affine = {}
        if not isinstance(moment_calibrated_affine, Mapping):
            raise ValueError(
                "cmpt.moment_calibrated_affine must be a mapping"
            )
        moment_calibrated_affine_enabled = bool(
            moment_calibrated_affine.get("enabled", False)
        )
        moment_calibrated_affine_uniform_ridge = float(
            moment_calibrated_affine.get("uniform_ridge", 1.0e-3)
        )
        moment_calibrated_affine_iterations = int(
            moment_calibrated_affine.get("max_iterations", 1000)
        )
        moment_calibrated_affine_ridge = float(
            moment_calibrated_affine.get("affine_ridge", affine_ridge)
        )
        if moment_calibrated_affine_uniform_ridge < 0.0:
            raise ValueError(
                "moment-calibrated affine uniform_ridge must be non-negative"
            )
        if moment_calibrated_affine_iterations <= 0:
            raise ValueError(
                "moment-calibrated affine max_iterations must be positive"
            )
        if moment_calibrated_affine_ridge <= 0.0:
            raise ValueError(
                "moment-calibrated affine affine_ridge must be positive"
            )
        if moment_calibrated_affine_enabled and transport != "affine_ridge":
            raise ValueError(
                "moment-calibrated comparison requires affine_ridge transport"
            )
        moment_transport = cmpt.get("moment_transport", {})
        if moment_transport is None:
            moment_transport = {}
        if not isinstance(moment_transport, Mapping):
            raise ValueError("cmpt.moment_transport must be a mapping")
        moment_transport_enabled = bool(
            moment_transport.get("enabled", False)
        )
        moment_transport_mode = str(
            moment_transport.get("mode", "low_rank_pca")
        ).lower()
        if moment_transport_mode not in {
            "low_rank_pca",
            "polynomial_kernel",
        }:
            raise ValueError(
                "moment transport mode must be low_rank_pca or "
                "polynomial_kernel"
            )
        moment_transport_rank = int(moment_transport.get("rank", 6))
        moment_transport_affine_ridge = float(
            moment_transport.get("affine_ridge", affine_ridge)
        )
        moment_transport_quadratic_ridge = float(
            moment_transport.get("quadratic_ridge", 1.0)
        )
        moment_transport_residual_covariance_scale = float(
            moment_transport.get("residual_covariance_scale", 0.0)
        )
        if moment_transport_rank <= 0:
            raise ValueError("moment transport rank must be positive")
        if (
            moment_transport_affine_ridge <= 0.0
            or moment_transport_quadratic_ridge <= 0.0
        ):
            raise ValueError("moment transport ridges must be positive")
        if moment_transport_residual_covariance_scale < 0.0:
            raise ValueError(
                "moment transport residual covariance scale must be non-negative"
            )
        moment_grid = moment_transport.get("grid", {})
        if moment_grid is None:
            moment_grid = {}
        if not isinstance(moment_grid, Mapping):
            raise ValueError("cmpt.moment_transport.grid must be a mapping")
        moment_grid_enabled = bool(moment_grid.get("enabled", False))

        def numeric_sequence(
            key: str,
            default: Sequence[int | float],
            cast: Callable[[Any], int | float],
        ) -> tuple[int | float, ...]:
            raw = moment_grid.get(key, default)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(
                    f"cmpt.moment_transport.grid.{key} must be a sequence"
                )
            values = tuple(cast(value) for value in raw)
            if not values or len(set(values)) != len(values):
                raise ValueError(
                    f"moment transport grid {key} must be non-empty and unique"
                )
            return values

        moment_grid_ranks = tuple(
            int(value)
            for value in numeric_sequence(
                "ranks", (2, 4, 6, 8, 12, 16), int
            )
        )
        moment_grid_quadratic_ridges = tuple(
            float(value)
            for value in numeric_sequence(
                "quadratic_ridges", (0.1, 1.0, 10.0), float
            )
        )
        moment_grid_correction_scales = tuple(
            float(value)
            for value in numeric_sequence(
                "correction_scales", (0.25, 0.5, 0.75, 1.0), float
            )
        )
        moment_grid_validation_folds = int(
            moment_grid.get("validation_folds", 5)
        )
        moment_grid_validation_fold = int(
            moment_grid.get("validation_fold", 0)
        )
        if any(value <= 0 for value in moment_grid_ranks):
            raise ValueError("moment transport grid ranks must be positive")
        if any(value <= 0.0 for value in moment_grid_quadratic_ridges):
            raise ValueError("moment transport grid ridges must be positive")
        if any(
            not 0.0 < value <= 1.0
            for value in moment_grid_correction_scales
        ):
            raise ValueError(
                "moment transport correction scales must lie in (0, 1]"
            )
        if moment_grid_validation_folds < 2:
            raise ValueError("moment grid validation_folds must be at least 2")
        if not 0 <= moment_grid_validation_fold < moment_grid_validation_folds:
            raise ValueError(
                "moment grid validation_fold must lie inside validation_folds"
            )
        if moment_grid_enabled and (
            not moment_transport_enabled
            or moment_transport_mode != "low_rank_pca"
        ):
            raise ValueError(
                "moment grid requires enabled low_rank_pca moment transport"
            )

        return cls(
            learner=str(experiment["learner"]),
            checkpoint_directory=project_path(experiment["checkpoints"]),
            output_file=project_path(experiment["output"]),
            expected_checkpoint_method=str(
                experiment["expected_checkpoint_method"]
            ).lower(),
            expected_sessions=int(experiment.get("expected_sessions", 11)),
            expected_exemplars_per_class=int(
                experiment.get("expected_exemplars_per_class", 20)
            ),
            expected_casper_enabled=bool(
                experiment.get("expected_casper_enabled", False)
            ),
            device=str(config.get("device", "cuda:0")),
            transport=transport,
            affine_ridge=affine_ridge,
            prototype_horizontal_flip=bool(
                cmpt.get("prototype_horizontal_flip", True)
            ),
            support_horizontal_flip=bool(
                cmpt.get("support_horizontal_flip", True)
            ),
            query_horizontal_flip=bool(
                cmpt.get("query_horizontal_flip", False)
            ),
            center_strength=center_strength,
            strict_parity=bool(cmpt.get("strict_parity", True)),
            parity_tolerance=parity_tolerance,
            cpu_threads=cpu_threads,
            prototype_interpolation_alphas=interpolation_alphas,
            adaptive_alpha_enabled=bool(
                adaptive_alpha.get("enabled", False)
            ),
            adaptive_alpha_folds=adaptive_alpha_folds,
            accuracy_oracle_enabled=accuracy_oracle_enabled,
            class_geometric_oracle_enabled=(
                class_geometric_oracle_enabled
            ),
            full_mean_oracle_enabled=full_mean_oracle_enabled,
            oracle_alpha_grid=oracle_alpha_grid,
            component_ablation_enabled=component_ablation_enabled,
            neighbor_affine_enabled=neighbor_affine_enabled,
            neighbor_affine_classes=neighbor_affine_classes,
            herding_extrapolation_enabled=herding_extrapolation_enabled,
            herding_full_prefix=herding_full_prefix,
            herding_reference_prefix=herding_reference_prefix,
            persistent_quadrature_enabled=(
                persistent_quadrature_enabled
            ),
            persistent_quadrature_ridge=(
                persistent_quadrature_ridge
            ),
            persistent_quadrature_iterations=(
                persistent_quadrature_iterations
            ),
            persistent_quadrature_multiview=(
                persistent_quadrature_multiview
            ),
            persistent_quadrature_previous_model_view=(
                persistent_quadrature_previous_model_view
            ),
            canonical_reference_enabled=canonical_reference_enabled,
            canonical_reference_ridge=canonical_reference_ridge,
            population_mass_transport_enabled=(
                population_mass_transport_enabled
            ),
            population_mass_transport_ridge=(
                population_mass_transport_ridge
            ),
            moment_calibrated_affine_enabled=(
                moment_calibrated_affine_enabled
            ),
            moment_calibrated_affine_uniform_ridge=(
                moment_calibrated_affine_uniform_ridge
            ),
            moment_calibrated_affine_iterations=(
                moment_calibrated_affine_iterations
            ),
            moment_calibrated_affine_ridge=(
                moment_calibrated_affine_ridge
            ),
            moment_transport_enabled=moment_transport_enabled,
            moment_transport_mode=moment_transport_mode,
            moment_transport_rank=moment_transport_rank,
            moment_transport_affine_ridge=(
                moment_transport_affine_ridge
            ),
            moment_transport_quadratic_ridge=(
                moment_transport_quadratic_ridge
            ),
            moment_transport_residual_covariance_scale=(
                moment_transport_residual_covariance_scale
            ),
            moment_grid_enabled=moment_grid_enabled,
            moment_grid_ranks=moment_grid_ranks,
            moment_grid_quadratic_ridges=(
                moment_grid_quadratic_ridges
            ),
            moment_grid_correction_scales=(
                moment_grid_correction_scales
            ),
            moment_grid_validation_folds=moment_grid_validation_folds,
            moment_grid_validation_fold=moment_grid_validation_fold,
        )


@dataclass(frozen=True)
class TrajectoryAudit:
    learner: str
    checkpoint_count: int
    session_ids: tuple[int, ...]
    protocol_id: str
    checkpoint_method: str
    evaluation_classifier: str
    feature_dim: int
    exemplars_per_class: int
    final_memory_classes: int
    old_exemplar_identities_stable: bool
    casper_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_ids"] = list(self.session_ids)
        return payload


@dataclass(frozen=True)
class NativeClassifierSpec:
    """Test-time classifier that belongs to a learner's native contract."""

    classifier: str
    implementation: str
    distinct_from_nme: bool
    uses_learned_parameters: bool
    query_horizontal_flip: bool = False
    scale: float | None = None
    epsilon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedTransportSupport:
    """Paired replay features for affine fitting and uncertainty estimates."""

    old_fit_features: Tensor
    current_fit_features: Tensor
    fit_targets: Tensor
    old_exemplar_features: Tensor
    current_exemplar_features: Tensor
    targets: Tensor

    @property
    def fit_support_count(self) -> int:
        return int(self.old_fit_features.shape[0])

    @property
    def exemplar_count(self) -> int:
        return int(self.old_exemplar_features.shape[0])


@dataclass(frozen=True)
class MomentGridState:
    means: Tensor
    second_moments: Tensor
    direct_bank: Tensor


@dataclass(frozen=True)
class MomentCalibratedAffineState:
    prototypes: Tensor
    means: Tensor
    second_moments: Tensor


def resolve_native_classifier(
    checkpoint: Mapping[str, Any],
) -> NativeClassifierSpec:
    """Resolve native inference without confusing a training-only FC with it.

    The common-recipe LUCIR adapter intentionally uses the iCaRL substrate
    name, so its normalized-cosine native head is identified from the saved
    feature-cosine configuration rather than from a learner-name string.
    """

    contract = _required_mapping(checkpoint, "method_contract")
    method = str(contract.get("name", "")).lower()
    config = _required_mapping(checkpoint, "config")
    method_config = config.get("method", {})
    if not isinstance(method_config, Mapping):
        raise ValueError("checkpoint config.method must be a mapping")

    if method == "icarl":
        feature_cosine = method_config.get(
            "feature_cosine_distillation", {}
        )
        if feature_cosine is None:
            feature_cosine = {}
        if not isinstance(feature_cosine, Mapping):
            raise ValueError(
                "method.feature_cosine_distillation must be a mapping"
            )
        if bool(feature_cosine.get("enabled", False)):
            training_classifier = str(
                feature_cosine.get("training_classifier", "")
            ).lower()
            if training_classifier != "normalized_cosine":
                raise ValueError(
                    "enabled LUCIR feature distillation requires its saved "
                    "normalized_cosine training classifier"
                )
            return NativeClassifierSpec(
                classifier="normalized_cosine_head",
                implementation="lucir_training_cosine_logits",
                distinct_from_nme=True,
                uses_learned_parameters=True,
                scale=float(feature_cosine.get("cosine_scale", 10.0)),
                epsilon=float(feature_cosine.get("epsilon", 1.0e-12)),
            )
        return NativeClassifierSpec(
            classifier="exemplar_nme",
            implementation="native_nme_alias",
            distinct_from_nme=False,
            uses_learned_parameters=False,
        )

    if method in {"casper", "sacil"}:
        return NativeClassifierSpec(
            classifier="exemplar_nme",
            implementation="native_nme_alias",
            distinct_from_nme=False,
            uses_learned_parameters=False,
        )

    if method == "create":
        return NativeClassifierSpec(
            classifier="classwise_reconstruction_error",
            implementation="model_forward_logits",
            distinct_from_nme=True,
            uses_learned_parameters=True,
        )

    if method in {
        "joint",
        "finetune",
        "replay",
        "podnet",
        "afc",
        "fgp",
        "cscct",
    }:
        return NativeClassifierSpec(
            classifier="learned_classification_head",
            implementation="model_forward_logits",
            distinct_from_nme=True,
            uses_learned_parameters=True,
        )

    raise ValueError(f"unsupported native classifier contract: {method}")


class _NativeClassifierView(nn.Module):
    """Expose exactly the saved learner's native logits to the evaluator."""

    def __init__(self, model: nn.Module, spec: NativeClassifierSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.eval()

    def forward(self, images: Tensor) -> Tensor:
        if self.spec.implementation == "model_forward_logits":
            logits = self.model(images)
        elif self.spec.implementation == "lucir_training_cosine_logits":
            classifier = getattr(self.model, "classifier", None)
            class_weights = getattr(classifier, "weight", None)
            if not isinstance(class_weights, Tensor):
                raise TypeError("LUCIR native evaluation requires FC weights")
            features = self.model.extract_features(images)
            logits = normalized_cosine_classifier_logits(
                features,
                class_weights,
                scale=float(self.spec.scale),
                epsilon=float(self.spec.epsilon),
            )
        else:
            raise RuntimeError(
                "NME aliases must not be sent through a native-head view"
            )
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise TypeError("native classifier must return a logits matrix")
        return logits


def _required_mapping(
    mapping: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration key {key!r} must be a mapping")
    return value


def discover_checkpoint_paths(
    checkpoint_directory: str | Path,
) -> list[Path]:
    directory = Path(checkpoint_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    indexed: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _CHECKPOINT_PATTERN.match(path.name)
        if path.is_file() and match is not None:
            indexed.append((int(match.group(1)), path.resolve()))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        raise FileNotFoundError(f"no session checkpoints found in {directory}")
    session_ids = [item[0] for item in indexed]
    if session_ids != list(range(len(session_ids))):
        raise ValueError(
            "checkpoint sequence must be contiguous from S0; found "
            f"{session_ids}"
        )
    return [item[1] for item in indexed]


def _memory_indices(checkpoint: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    memory = _required_mapping(checkpoint, "memory")
    raw = _required_mapping(memory, "indices")
    return {
        int(class_id): tuple(int(index) for index in indices)
        for class_id, indices in raw.items()
    }


def _checkpoint_session_metric(
    checkpoint: Mapping[str, Any], session_id: int
) -> Mapping[str, Any]:
    metrics = _required_mapping(checkpoint, "metrics")
    records = metrics.get("records")
    if not isinstance(records, Sequence):
        raise ValueError("checkpoint metrics.records must be a sequence")
    for record in records:
        if isinstance(record, Mapping) and int(record["session_id"]) == session_id:
            return record
    raise ValueError(f"checkpoint lacks stored metric for session {session_id}")


def audit_checkpoint_trajectory(
    checkpoints: Sequence[Mapping[str, Any]],
    settings: CMPTExperimentSettings,
) -> TrajectoryAudit:
    if len(checkpoints) != settings.expected_sessions:
        raise ValueError(
            f"{settings.learner}: expected {settings.expected_sessions} "
            f"checkpoints, found {len(checkpoints)}"
        )
    session_ids = tuple(int(value["session_id"]) for value in checkpoints)
    if session_ids != tuple(range(len(checkpoints))):
        raise ValueError(
            f"{settings.learner}: non-contiguous session IDs {session_ids}"
        )

    protocols: set[str] = set()
    methods: set[str] = set()
    classifiers: set[str] = set()
    dimensions: set[int] = set()
    casper_flags: set[bool] = set()
    prototype_flip_flags: set[bool] = set()
    previous_memory: dict[int, tuple[int, ...]] | None = None
    stable = True

    for session_id, checkpoint in enumerate(checkpoints):
        protocols.add(str(checkpoint.get("protocol_id", "")))
        contract = _required_mapping(checkpoint, "method_contract")
        methods.add(str(contract.get("name", "")).lower())
        classifiers.add(str(contract.get("evaluation_classifier", "")).lower())
        means = checkpoint.get("class_means")
        if not isinstance(means, Tensor) or means.ndim != 2:
            raise ValueError(
                f"{settings.learner} S{session_id}: class_means is missing"
            )
        if not bool(torch.isfinite(means).all()):
            raise ValueError(
                f"{settings.learner} S{session_id}: non-finite class means"
            )
        dimensions.add(int(means.shape[1]))
        memory = _memory_indices(checkpoint)
        memory_state = _required_mapping(checkpoint, "memory")
        if int(memory_state["exemplars_per_class"]) != (
            settings.expected_exemplars_per_class
        ):
            raise ValueError(
                f"{settings.learner} S{session_id}: memory limit does not "
                "match the expected exemplar count"
            )
        if len(memory) != int(means.shape[0]):
            raise ValueError(
                f"{settings.learner} S{session_id}: memory/class-mean count "
                f"mismatch ({len(memory)} vs {means.shape[0]})"
            )
        counts = {len(indices) for indices in memory.values()}
        if counts != {settings.expected_exemplars_per_class}:
            raise ValueError(
                f"{settings.learner} S{session_id}: expected "
                f"{settings.expected_exemplars_per_class} exemplars/class, "
                f"found {sorted(counts)}"
            )
        if previous_memory is not None:
            for class_id, indices in previous_memory.items():
                if memory.get(class_id) != indices:
                    stable = False
        previous_memory = memory
        casper = checkpoint.get("casper_options", {})
        casper_flags.add(
            bool(casper.get("enabled", False))
            if isinstance(casper, Mapping)
            else False
        )
        checkpoint_config = _required_mapping(checkpoint, "config")
        evaluation = _required_mapping(checkpoint_config, "evaluation")
        prototype_flip_flags.add(bool(evaluation.get("horizontal_flip", False)))
        _checkpoint_session_metric(checkpoint, session_id)

    if len(protocols) != 1 or "" in protocols:
        raise ValueError(f"{settings.learner}: inconsistent protocol IDs")
    if methods != {settings.expected_checkpoint_method}:
        raise ValueError(
            f"{settings.learner}: expected checkpoint method "
            f"{settings.expected_checkpoint_method!r}, found {sorted(methods)}"
        )
    if classifiers != {"nme"}:
        raise ValueError(
            f"{settings.learner}: CMPT requires NME checkpoints, found "
            f"{sorted(classifiers)}"
        )
    if len(dimensions) != 1:
        raise ValueError(
            f"{settings.learner}: feature dimensions change across sessions: "
            f"{sorted(dimensions)}"
        )
    if prototype_flip_flags != {settings.prototype_horizontal_flip}:
        raise ValueError(
            f"{settings.learner}: checkpoint NME prototype flip convention "
            f"{sorted(prototype_flip_flags)} does not match CMPT setting "
            f"{settings.prototype_horizontal_flip}"
        )
    if not stable:
        raise ValueError(
            f"{settings.learner}: old exemplar identities change across sessions"
        )
    if len(casper_flags) != 1:
        raise ValueError(
            f"{settings.learner}: CaSpeR flag changes across sessions: "
            f"{sorted(casper_flags)}"
        )
    casper_enabled = casper_flags == {True}
    if settings.expected_casper_enabled != casper_enabled:
        raise ValueError(
            f"{settings.learner}: expected_casper_enabled="
            f"{settings.expected_casper_enabled}, observed {sorted(casper_flags)}"
        )
    assert previous_memory is not None
    return TrajectoryAudit(
        learner=settings.learner,
        checkpoint_count=len(checkpoints),
        session_ids=session_ids,
        protocol_id=next(iter(protocols)),
        checkpoint_method=next(iter(methods)),
        evaluation_classifier="nme",
        feature_dim=next(iter(dimensions)),
        exemplars_per_class=settings.expected_exemplars_per_class,
        final_memory_classes=len(previous_memory),
        old_exemplar_identities_stable=stable,
        casper_enabled=casper_enabled,
    )


def _load_model(
    trainer: UnifiedTable1Trainer,
    checkpoint: Mapping[str, Any],
    session_id: int,
) -> nn.Module:
    seen_classes = trainer.protocol.session(int(session_id)).stop
    base_classes = trainer.protocol.session(0).stop

    # Some incremental classifiers cannot be reconstructed by passing the
    # final class count to their constructor.  PODNet stores one consolidated
    # old chunk plus the latest chunk, while CSCCT stores every task chunk and
    # creates its scale-shift branch at the first expansion.  Replaying only
    # these parameter-free architecture transitions yields the exact module
    # graph expected by the checkpoint; it does not run training or use data.
    if trainer.method == "podnet" and session_id > 0:
        model = trainer._new_model(base_classes).to(trainer.device)
        for step in range(1, int(session_id) + 1):
            model.expand_classes(trainer.protocol.session(step).stop)
    elif trainer.method == "cscct" and session_id > 0:
        model = trainer._new_model(base_classes).to(trainer.device)
        for step in range(1, int(session_id) + 1):
            increment = trainer.protocol.session(step).size
            dummy = torch.zeros(
                increment,
                int(model.feature_dim),
                device=trainer.device,
            )
            model.expand_classes(dummy)
    else:
        model = trainer._new_model(int(seen_classes)).to(trainer.device)
    expected_type = str(checkpoint.get("model_type", ""))
    if expected_type and type(model).__name__ != expected_type:
        raise TypeError(
            f"checkpoint expects {expected_type}, reconstructed "
            f"{type(model).__name__}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _full_introduction_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
) -> Tensor:
    class_ids = trainer.protocol.classes_for_session(session_id)
    dataset = trainer.data.train_eval_dataset_for_classes(
        class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 13000
    )
    regular = collect_features(model, loader, trainer.device)
    features = regular.features
    targets = regular.original_targets
    if horizontal_flip:
        flipped = collect_features(
            model, loader, trainer.device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError("full-prototype flip rows are misaligned")
        features = torch.cat([features, flipped.features], dim=0)
        targets = torch.cat(
            [targets, flipped.original_targets], dim=0
        )
    return compute_prototypes(features, targets, class_ids).cpu()


def _full_current_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    class_ids: Sequence[int],
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, int]:
    """Oracle-only current means from every requested training image."""

    if not class_ids:
        raise ValueError("full-mean oracle requires at least one class")
    dataset = trainer.data.train_eval_dataset_for_classes(
        class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 15000
    )
    regular = collect_features(model, loader, trainer.device)
    features = regular.features
    targets = regular.original_targets
    if horizontal_flip:
        flipped = collect_features(
            model, loader, trainer.device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError("geometric-oracle flip rows are misaligned")
        features = torch.cat([features, flipped.features], dim=0)
        targets = torch.cat(
            [targets, flipped.original_targets], dim=0
        )
    return (
        compute_prototypes(features, targets, class_ids).cpu(),
        len(dataset),
    )


def _full_current_old_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, int]:
    """Oracle-only current means from all old-class training images."""

    class_ids = trainer.protocol.old_classes(session_id)
    if not class_ids:
        raise ValueError("class-geometric oracle requires old classes")
    return _full_current_prototypes(
        trainer,
        model,
        session_id,
        class_ids,
        horizontal_flip=horizontal_flip,
    )


def _herding_extrapolated_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
    full_prefix: int,
    reference_prefix: int,
) -> tuple[Tensor, Tensor, int]:
    """Recompute ordered-memory NME and its Richardson extrapolation."""

    expected_indices = trainer.memory.all_indices(
        trainer.protocol.class_order
    )
    loader = trainer._memory_loader(session_id, augment=False)
    regular = collect_features(model, loader, trainer.device)
    expected = torch.tensor(expected_indices, dtype=torch.long)
    if not torch.equal(regular.indices.cpu().long(), expected):
        raise RuntimeError(
            "memory feature rows do not preserve stored herding order"
        )
    contributions = F.normalize(regular.features.float(), dim=1)
    if horizontal_flip:
        flipped = collect_features(
            model, loader, trainer.device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError("herding flip feature rows are misaligned")
        contributions = 0.5 * (
            contributions + F.normalize(flipped.features.float(), dim=1)
        )
    ordinary, extrapolated = herding_prefix_extrapolated_means(
        contributions,
        regular.targets,
        trainer.protocol.session(session_id).stop,
        full_prefix=full_prefix,
        reference_prefix=reference_prefix,
    )
    return ordinary.cpu(), extrapolated.cpu(), len(expected_indices)


def _nme_contributions(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collect one matched NME contribution per original image."""

    views, targets, original_targets, indices = _normalized_feature_views(
        model,
        loader,
        device,
        horizontal_flip=horizontal_flip,
    )
    return views.mean(dim=0), targets, original_targets, indices


def _normalized_feature_views(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return separate normalized deterministic views as [V,N,D]."""

    regular = collect_features(model, loader, device)
    contributions = [F.normalize(regular.features.float(), dim=1)]
    if horizontal_flip:
        flipped = collect_features(
            model, loader, device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError("quadrature flip feature rows are misaligned")
        contributions.append(F.normalize(flipped.features.float(), dim=1))
    return (
        torch.stack(contributions).cpu(),
        regular.targets.detach().cpu().long(),
        regular.original_targets.detach().cpu().long(),
        regular.indices.detach().cpu().long(),
    )


def _full_introduction_moments(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, Tensor, int]:
    original_class_ids = trainer.protocol.classes_for_session(session_id)
    dataset = trainer.data.train_eval_dataset_for_classes(
        original_class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 18000
    )
    features, _, original_targets, _ = _nme_contributions(
        model,
        loader,
        trainer.device,
        horizontal_flip=horizontal_flip,
    )
    means, seconds = class_first_second_moments(
        features, original_targets, original_class_ids
    )
    return means.cpu(), seconds.cpu(), len(dataset)


def _full_introduction_statistics(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Collect introduction features once for prototypes and moments."""

    original_class_ids = trainer.protocol.classes_for_session(session_id)
    dataset = trainer.data.train_eval_dataset_for_classes(
        original_class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 19000
    )
    regular = collect_features(model, loader, trainer.device)
    prototype_features = regular.features
    prototype_targets = regular.original_targets
    contributions = F.normalize(regular.features.float(), dim=1)
    original_targets = regular.original_targets.detach().cpu().long()
    if horizontal_flip:
        flipped = collect_features(
            model, loader, trainer.device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError(
                "shared introduction flip rows are misaligned"
            )
        prototype_features = torch.cat(
            [prototype_features, flipped.features], dim=0
        )
        prototype_targets = torch.cat(
            [prototype_targets, flipped.original_targets], dim=0
        )
        contributions = 0.5 * (
            contributions
            + F.normalize(flipped.features.float(), dim=1)
        )
    prototypes = compute_prototypes(
        prototype_features, prototype_targets, original_class_ids
    )
    means, seconds = class_first_second_moments(
        contributions, original_targets, original_class_ids
    )
    return prototypes.cpu(), means.cpu(), seconds.cpu(), len(dataset)


def _paired_moment_support(
    trainer: UnifiedTable1Trainer,
    previous_model: nn.Module,
    current_model: nn.Module,
    previous_session_id: int,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, Tensor, int]:
    loader = trainer._memory_loader(previous_session_id, augment=False)
    old, old_targets, _, old_indices = _nme_contributions(
        previous_model,
        loader,
        trainer.device,
        horizontal_flip=horizontal_flip,
    )
    current, current_targets, _, current_indices = _nme_contributions(
        current_model,
        loader,
        trainer.device,
        horizontal_flip=horizontal_flip,
    )
    if not (
        torch.equal(old_indices, current_indices)
        and torch.equal(old_targets, current_targets)
    ):
        raise RuntimeError("moment-transport support rows are misaligned")
    return old, current, int(old.shape[0])


def _moment_support_from_paired(
    support: PairedTransportSupport,
) -> tuple[Tensor, Tensor, int]:
    """Reuse affine feature rows as per-exemplar moment contributions."""

    exemplar_count = support.exemplar_count

    def contributions(values: Tensor) -> Tensor:
        normalized = F.normalize(values.detach().cpu().float(), dim=1)
        if normalized.shape[0] == exemplar_count:
            return normalized
        if normalized.shape[0] == 2 * exemplar_count:
            return 0.5 * (
                normalized[:exemplar_count]
                + normalized[exemplar_count:]
            )
        raise ValueError(
            "affine support rows do not align with exemplar identities"
        )

    return (
        contributions(support.old_fit_features),
        contributions(support.current_fit_features),
        exemplar_count,
    )


def _moment_grid_key(rank: int, ridge: float, scale: float) -> str:
    return (
        f"r{int(rank)}_q{format(float(ridge), '.6g')}"
        f"_g{format(float(scale), '.6g')}"
    )


def _moment_grid_specs(
    ranks: Sequence[int],
    ridges: Sequence[float],
    scales: Sequence[float],
) -> tuple[tuple[str, int, float, float], ...]:
    return tuple(
        (
            _moment_grid_key(rank, ridge, scale),
            int(rank),
            float(ridge),
            float(scale),
        )
        for rank in ranks
        for ridge in ridges
        for scale in scales
    )


def _stratified_moment_grid_split(
    targets: Tensor,
    *,
    folds: int,
    validation_fold: int,
) -> tuple[Tensor, Tensor]:
    labels = targets.detach().cpu().long().flatten()
    fit_indices: list[Tensor] = []
    validation_indices: list[Tensor] = []
    for class_id in labels.unique(sorted=True).tolist():
        indices = torch.nonzero(labels == int(class_id)).flatten()
        mask = (
            torch.arange(indices.numel(), dtype=torch.long) % int(folds)
            == int(validation_fold)
        )
        if int(mask.sum().item()) == 0 or int((~mask).sum().item()) == 0:
            raise ValueError(
                f"class {class_id} cannot be split into {folds} folds"
            )
        validation_indices.append(indices[mask])
        fit_indices.append(indices[~mask])
    return torch.cat(fit_indices), torch.cat(validation_indices)


def _moment_grid_validation_error(
    predicted_prototypes: Tensor,
    current_validation_features: Tensor,
    validation_targets: Tensor,
) -> dict[str, float]:
    class_ids = validation_targets.detach().cpu().long().unique(sorted=True)
    observed = compute_prototypes(
        current_validation_features.detach().cpu().float(),
        validation_targets.detach().cpu().long(),
        class_ids,
    )
    predicted = F.normalize(predicted_prototypes.detach().cpu().float(), dim=1)
    if predicted.shape != observed.shape:
        raise ValueError("moment-grid validation prototype shapes differ")
    features = F.normalize(
        current_validation_features.detach().cpu().float(), dim=1
    )
    targets = validation_targets.detach().cpu().long()
    logits = features @ predicted.T
    predictions = logits.argmax(dim=1)
    true_scores = logits.gather(1, targets[:, None]).squeeze(1)
    competing = logits.clone()
    competing.scatter_(1, targets[:, None], float("-inf"))
    margins = true_scores - competing.max(dim=1).values
    return {
        "prototype_cosine_distance": float(
            (1.0 - (predicted * observed).sum(dim=1)).mean().item()
        ),
        "nme_accuracy": float(predictions.eq(targets).float().mean().item()),
        "nme_cosine_margin": float(margins.mean().item()),
    }


def _fit_new_persistent_quadrature_weights(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
    uniform_ridge: float,
    max_iterations: int,
    multiview: bool,
    previous_model: nn.Module | None,
    previous_model_view: bool,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Fit weights for classes whose full training set is currently legal."""

    original_class_ids = trainer.protocol.classes_for_session(session_id)
    full_dataset = trainer.data.train_eval_dataset_for_classes(
        original_class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    full_loader = trainer._loader(
        full_dataset, shuffle=False, session_id=session_id + 17000
    )
    full_views, _, full_original, full_indices = _normalized_feature_views(
        model,
        full_loader,
        trainer.device,
        horizontal_flip=horizontal_flip,
    )

    selected_indices: list[int] = []
    for class_id in original_class_ids:
        class_indices = trainer.memory.indices_for_class(class_id)
        if len(class_indices) != trainer.memory.exemplars_per_class:
            raise RuntimeError(
                f"class {class_id} lacks a complete quadrature support"
            )
        selected_indices.extend(class_indices)
    memory_dataset = trainer.data.train_eval_dataset_from_indices(
        selected_indices,
        is_replay=True,
    )
    memory_loader = trainer._loader(
        memory_dataset, shuffle=False, session_id=session_id + 17100
    )
    memory_views, _, memory_original, memory_indices = (
        _normalized_feature_views(
        model,
        memory_loader,
        trainer.device,
        horizontal_flip=horizontal_flip,
        )
    )
    expected = torch.tensor(selected_indices, dtype=torch.long)
    if not torch.equal(memory_indices, expected):
        raise RuntimeError(
            "quadrature support does not preserve stored exemplar order"
        )
    if not set(memory_indices.tolist()).issubset(set(full_indices.tolist())):
        raise RuntimeError("quadrature exemplars are absent from full data")

    if not multiview:
        full_views = full_views.mean(dim=0, keepdim=True)
        memory_views = memory_views.mean(dim=0, keepdim=True)
    if previous_model_view and previous_model is not None:
        previous_full, _, previous_full_original, previous_full_indices = (
            _normalized_feature_views(
                previous_model,
                full_loader,
                trainer.device,
                horizontal_flip=horizontal_flip,
            )
        )
        previous_memory, _, previous_memory_original, previous_memory_indices = (
            _normalized_feature_views(
                previous_model,
                memory_loader,
                trainer.device,
                horizontal_flip=horizontal_flip,
            )
        )
        if not (
            torch.equal(previous_full_original, full_original)
            and torch.equal(previous_full_indices, full_indices)
            and torch.equal(previous_memory_original, memory_original)
            and torch.equal(previous_memory_indices, memory_indices)
        ):
            raise RuntimeError(
                "previous/current quadrature views are not image-aligned"
            )
        full_views = torch.cat([previous_full, full_views], dim=0)
        memory_views = torch.cat([previous_memory, memory_views], dim=0)

    fitted: list[Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for class_id in original_class_ids:
        class_memory = memory_views[:, memory_original == int(class_id)]
        class_full = full_views[:, full_original == int(class_id)]
        weights, stats = fit_multiview_persistent_quadrature_weights(
            class_memory,
            class_full,
            uniform_ridge=uniform_ridge,
            max_iterations=max_iterations,
        )
        fitted.append(weights)
        diagnostics.append(
            {
                "original_class_id": int(class_id),
                "population_count": int(class_full.shape[1]),
                "exemplar_count": int(class_memory.shape[1]),
                "weights": [float(value) for value in weights.tolist()],
                **stats,
            }
        )
    return torch.stack(fitted), diagnostics


def _persistent_quadrature_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    class_weights: Tensor,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, int]:
    expected_indices = trainer.memory.all_indices(
        trainer.protocol.class_order
    )
    loader = trainer._memory_loader(session_id, augment=False)
    features, targets, _, indices = _nme_contributions(
        model,
        loader,
        trainer.device,
        horizontal_flip=horizontal_flip,
    )
    expected = torch.tensor(expected_indices, dtype=torch.long)
    if not torch.equal(indices, expected):
        raise RuntimeError(
            "current quadrature features do not preserve exemplar identity"
        )
    return (
        weighted_class_prototypes(features, targets, class_weights).cpu(),
        len(expected_indices),
    )


def _quadrature_fit_support_weights(
    class_weights: Tensor,
    fit_support_count: int,
) -> Tensor:
    """Align per-exemplar masses with original[/flip] support rows."""

    base = class_weights.detach().float().reshape(-1)
    count = int(fit_support_count)
    if count == base.numel():
        return base
    if count == 2 * base.numel():
        # Paired support concatenates every original row and then every flip.
        return torch.cat([base, base], dim=0)
    raise ValueError(
        "quadrature weights do not align with affine fit support rows"
    )


def _paired_support_features(
    trainer: UnifiedTable1Trainer,
    previous_model: nn.Module,
    current_model: nn.Module,
    previous_session_id: int,
    *,
    horizontal_flip: bool,
) -> PairedTransportSupport:
    loader = trainer._memory_loader(previous_session_id, augment=False)
    old = collect_features(previous_model, loader, trainer.device)
    current = collect_features(current_model, loader, trainer.device)
    if not torch.equal(old.indices, current.indices):
        raise RuntimeError("old/current transport support rows are misaligned")
    if not torch.equal(old.targets, current.targets):
        raise RuntimeError("old/current transport support labels are misaligned")
    old_features = old.features
    current_features = current.features
    fit_targets = current.targets.detach().cpu().long()
    old_exemplar_features = F.normalize(old.features.float(), dim=1)
    current_exemplar_features = F.normalize(current.features.float(), dim=1)
    if horizontal_flip:
        old_flip = collect_features(
            previous_model, loader, trainer.device, horizontal_flip=True
        )
        current_flip = collect_features(
            current_model, loader, trainer.device, horizontal_flip=True
        )
        if not (
            torch.equal(old.indices, old_flip.indices)
            and torch.equal(old.indices, current_flip.indices)
        ):
            raise RuntimeError("transport-support flip rows are misaligned")
        old_features = torch.cat([old_features, old_flip.features], dim=0)
        current_features = torch.cat(
            [current_features, current_flip.features], dim=0
        )
        fit_targets = torch.cat([fit_targets, fit_targets], dim=0)
        old_exemplar_features = F.normalize(
            old_exemplar_features
            + F.normalize(old_flip.features.float(), dim=1),
            dim=1,
        )
        current_exemplar_features = F.normalize(
            current_exemplar_features
            + F.normalize(current_flip.features.float(), dim=1),
            dim=1,
        )
    return PairedTransportSupport(
        old_fit_features=old_features,
        current_fit_features=current_features,
        fit_targets=fit_targets,
        old_exemplar_features=old_exemplar_features,
        current_exemplar_features=current_exemplar_features,
        targets=current.targets.detach().cpu().long(),
    )


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    cmpt = [float(record["cmpt"]["accuracy"]) for record in records]
    native = [float(record["native"]["accuracy"]) for record in records]
    parity = [abs(float(record["parity_error"])) for record in records]
    incremental_baseline = baseline[1:] if len(baseline) > 1 else baseline
    incremental_cmpt = cmpt[1:] if len(cmpt) > 1 else cmpt
    incremental_native = native[1:] if len(native) > 1 else native
    baseline_aia = sum(baseline) / len(baseline)
    cmpt_aia = sum(cmpt) / len(cmpt)
    native_aia = sum(native) / len(native)
    return {
        "native_aia_percent": 100.0 * native_aia,
        "baseline_aia_percent": 100.0 * baseline_aia,
        "cmpt_aia_percent": 100.0 * cmpt_aia,
        "aia_delta_percent_points": 100.0 * (cmpt_aia - baseline_aia),
        "nme_minus_native_aia_percent_points": 100.0
        * (baseline_aia - native_aia),
        "cmpt_minus_native_aia_percent_points": 100.0
        * (cmpt_aia - native_aia),
        "native_incremental_aia_percent": 100.0
        * sum(incremental_native)
        / len(incremental_native),
        "baseline_incremental_aia_percent": 100.0
        * sum(incremental_baseline)
        / len(incremental_baseline),
        "cmpt_incremental_aia_percent": 100.0
        * sum(incremental_cmpt)
        / len(incremental_cmpt),
        "native_final_percent": 100.0 * native[-1],
        "baseline_final_percent": 100.0 * baseline[-1],
        "cmpt_final_percent": 100.0 * cmpt[-1],
        "final_delta_percent_points": 100.0 * (cmpt[-1] - baseline[-1]),
        "nme_minus_native_final_percent_points": 100.0
        * (baseline[-1] - native[-1]),
        "cmpt_minus_native_final_percent_points": 100.0
        * (cmpt[-1] - native[-1]),
        "max_parity_error_percent_points": 100.0 * max(parity),
    }


def _aggregate_component_ablation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate matched NME/global/class/combined prototype estimators."""

    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    global_affine = [float(record["cmpt"]["accuracy"]) for record in records]
    class_translation = [
        float(record["class_translation"]["accuracy"]) for record in records
    ]
    combined = [
        float(record["combined_cmpt"]["accuracy"]) for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    global_aia = sum(global_affine) / len(global_affine)
    class_aia = sum(class_translation) / len(class_translation)
    combined_aia = sum(combined) / len(combined)

    def metrics(values: Sequence[float]) -> dict[str, float]:
        incremental = values[1:] if len(values) > 1 else values
        aia = sum(values) / len(values)
        return {
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
        }

    class_metrics = metrics(class_translation)
    class_metrics.update(
        {
            "aia_delta_vs_nme_percent_points": 100.0
            * (class_aia - baseline_aia),
            "aia_delta_vs_global_percent_points": 100.0
            * (class_aia - global_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (class_translation[-1] - baseline[-1]),
            "final_delta_vs_global_percent_points": 100.0
            * (class_translation[-1] - global_affine[-1]),
        }
    )
    combined_metrics = metrics(combined)
    combined_metrics.update(
        {
            "aia_delta_vs_nme_percent_points": 100.0
            * (combined_aia - baseline_aia),
            "aia_delta_vs_global_percent_points": 100.0
            * (combined_aia - global_aia),
            "aia_delta_vs_class_percent_points": 100.0
            * (combined_aia - class_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (combined[-1] - baseline[-1]),
            "final_delta_vs_global_percent_points": 100.0
            * (combined[-1] - global_affine[-1]),
            "final_delta_vs_class_percent_points": 100.0
            * (combined[-1] - class_translation[-1]),
        }
    )
    return {
        "nme": metrics(baseline),
        "global_affine": metrics(global_affine),
        "class_translation": class_metrics,
        "combined_cmpt": combined_metrics,
    }


def _aggregate_neighbor_affine(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Aggregate matched local-neighborhood affine classification results."""

    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    global_affine = [float(record["cmpt"]["accuracy"]) for record in records]
    local_affine = [
        float(record["neighbor_affine"]["accuracy"]) for record in records
    ]
    incremental = local_affine[1:] if len(local_affine) > 1 else local_affine
    local_aia = sum(local_affine) / len(local_affine)
    baseline_aia = sum(baseline) / len(baseline)
    global_aia = sum(global_affine) / len(global_affine)
    return {
        "aia_percent": 100.0 * local_aia,
        "incremental_aia_percent": 100.0
        * sum(incremental)
        / len(incremental),
        "final_percent": 100.0 * local_affine[-1],
        "aia_delta_vs_nme_percent_points": 100.0
        * (local_aia - baseline_aia),
        "aia_delta_vs_global_percent_points": 100.0
        * (local_aia - global_aia),
        "final_delta_vs_nme_percent_points": 100.0
        * (local_affine[-1] - baseline[-1]),
        "final_delta_vs_global_percent_points": 100.0
        * (local_affine[-1] - global_affine[-1]),
    }


def _aggregate_interpolation(
    records: Sequence[Mapping[str, Any]],
    alphas: Sequence[float],
) -> dict[str, dict[str, float]]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    baseline_final = baseline[-1]
    result: dict[str, dict[str, float]] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        values = [
            float(record["prototype_interpolation"][key]["accuracy"])
            for record in records
        ]
        incremental = values[1:] if len(values) > 1 else values
        aia = sum(values) / len(values)
        result[key] = {
            "alpha": float(alpha),
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (values[-1] - baseline_final),
        }
    return result


def _aggregate_adaptive_alpha(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    baseline_final = baseline[-1]
    result: dict[str, dict[str, float]] = {}
    for mode in ("session", "raw_class", "shrunken_class"):
        values = [
            float(record["adaptive_alpha"][mode]["metrics"]["accuracy"])
            for record in records
        ]
        incremental = values[1:] if len(values) > 1 else values
        alpha_means = [
            float(record["adaptive_alpha"][mode]["alpha_mean"])
            for record in records[1:]
        ]
        aia = sum(values) / len(values)
        result[mode] = {
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (values[-1] - baseline_final),
            "mean_incremental_alpha": (
                sum(alpha_means) / len(alpha_means)
                if alpha_means
                else 0.0
            ),
        }
    return result


def _aggregate_accuracy_oracle(
    records: Sequence[Mapping[str, Any]],
    alphas: Sequence[float],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    pure = [float(record["cmpt"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    pure_aia = sum(pure) / len(pure)

    trajectories: dict[float, list[float]] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        trajectories[float(alpha)] = [
            float(record["prototype_interpolation"][key]["accuracy"])
            for record in records
        ]
    global_alpha, global_values = max(
        trajectories.items(),
        key=lambda item: (sum(item[1]) / len(item[1]), item[0]),
    )
    global_aia = sum(global_values) / len(global_values)

    session_alphas: list[float] = []
    session_values: list[float] = []
    for index in range(len(records)):
        alpha, value = max(
            (
                (alpha, values[index])
                for alpha, values in trajectories.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        session_alphas.append(float(alpha))
        session_values.append(float(value))
    session_aia = sum(session_values) / len(session_values)
    return {
        "learner_global": {
            "alpha": global_alpha,
            "aia_percent": 100.0 * global_aia,
            "final_percent": 100.0 * global_values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (global_aia - baseline_aia),
            "aia_delta_vs_pure_cmpt_percent_points": 100.0
            * (global_aia - pure_aia),
        },
        "session": {
            "alphas": session_alphas,
            "mean_incremental_alpha": (
                sum(session_alphas[1:]) / len(session_alphas[1:])
                if len(session_alphas) > 1
                else session_alphas[0]
            ),
            "aia_percent": 100.0 * session_aia,
            "final_percent": 100.0 * session_values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (session_aia - baseline_aia),
            "aia_delta_vs_pure_cmpt_percent_points": 100.0
            * (session_aia - pure_aia),
        },
        "selection_uses_test_accuracy": True,
        "oracle_only": True,
    }


def _aggregate_class_geometric_oracle(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    pure = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["class_geometric_oracle"]["metrics"]["accuracy"])
        for record in records
    ]
    alpha_means = [
        float(record["class_geometric_oracle"]["alpha_mean"])
        for record in records[1:]
    ]
    baseline_aia = sum(baseline) / len(baseline)
    pure_aia = sum(pure) / len(pure)
    aia = sum(values) / len(values)
    return {
        "aia_percent": 100.0 * aia,
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0 * (aia - baseline_aia),
        "aia_delta_vs_pure_cmpt_percent_points": 100.0
        * (aia - pure_aia),
        "mean_incremental_alpha": (
            sum(alpha_means) / len(alpha_means) if alpha_means else 0.0
        ),
        "uses_full_old_training_data": True,
        "oracle_only": True,
    }


def _aggregate_full_mean_oracle(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the direct full-training-data NME upper bounds."""

    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    old_only = [
        float(record["full_mean_oracle"]["old_only"]["accuracy"])
        for record in records
    ]
    all_seen = [
        float(record["full_mean_oracle"]["all_seen"]["accuracy"])
        for record in records
    ]

    def metrics(values: Sequence[float]) -> dict[str, float]:
        incremental = values[1:] if len(values) > 1 else values
        aia = sum(values) / len(values)
        baseline_aia = sum(baseline) / len(baseline)
        return {
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (values[-1] - baseline[-1]),
        }

    incremental_records = list(records[1:])
    old_gaps = [
        float(record["full_mean_oracle"]["diagnostics"]["old_mean_cosine_distance"])
        for record in incremental_records
    ]
    all_gaps = [
        float(record["full_mean_oracle"]["diagnostics"]["all_seen_mean_cosine_distance"])
        for record in records
    ]
    return {
        "old_only": metrics(old_only),
        "all_seen": metrics(all_seen),
        "mean_incremental_old_prototype_cosine_distance": (
            sum(old_gaps) / len(old_gaps) if old_gaps else 0.0
        ),
        "mean_all_seen_prototype_cosine_distance": (
            sum(all_gaps) / len(all_gaps) if all_gaps else 0.0
        ),
        "uses_full_training_data": True,
        "uses_test_labels_for_selection": False,
        "oracle_only": True,
    }


def _aggregate_herding_extrapolation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    cmpt = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["herding_extrapolation"]["accuracy"])
        for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    cmpt_aia = sum(cmpt) / len(cmpt)
    aia = sum(values) / len(values)
    incremental = values[1:] if len(values) > 1 else values
    parity = [
        float(record["herding_extrapolation_diagnostics"]["nme_parity_max_abs"])
        for record in records
    ]
    return {
        "aia_percent": 100.0 * aia,
        "incremental_aia_percent": 100.0
        * sum(incremental)
        / len(incremental),
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0
        * (aia - baseline_aia),
        "aia_delta_vs_cmpt_percent_points": 100.0 * (aia - cmpt_aia),
        "final_delta_vs_nme_percent_points": 100.0
        * (values[-1] - baseline[-1]),
        "max_nme_prototype_parity_error": max(parity),
        "uses_test_data_for_selection": False,
        "training_reused_without_changes": True,
    }


def _aggregate_persistent_quadrature(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    cmpt = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["persistent_quadrature"]["accuracy"])
        for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    cmpt_aia = sum(cmpt) / len(cmpt)
    aia = sum(values) / len(values)
    incremental = values[1:] if len(values) > 1 else values
    introduced = [
        diagnostic
        for record in records
        for diagnostic in record[
            "persistent_quadrature_diagnostics"
        ]["introduced_classes"]
    ]
    return {
        "aia_percent": 100.0 * aia,
        "incremental_aia_percent": 100.0
        * sum(incremental)
        / len(incremental),
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0
        * (aia - baseline_aia),
        "aia_delta_vs_cmpt_percent_points": 100.0 * (aia - cmpt_aia),
        "final_delta_vs_nme_percent_points": 100.0
        * (values[-1] - baseline[-1]),
        "mean_introduction_uniform_cosine_distance": sum(
            float(value["uniform_target_cosine_distance"])
            for value in introduced
        )
        / len(introduced),
        "mean_introduction_weighted_cosine_distance": sum(
            float(value["weighted_target_cosine_distance"])
            for value in introduced
        )
        / len(introduced),
        "mean_effective_sample_size": sum(
            float(value["effective_sample_size"])
            for value in introduced
        )
        / len(introduced),
        "uses_test_data_for_selection": False,
        "uses_only_class_introduction_full_data": True,
        "training_reused_without_changes": True,
    }


def _aggregate_canonical_reference(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    sequential = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["canonical_reference"]["accuracy"])
        for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    sequential_aia = sum(sequential) / len(sequential)
    aia = sum(values) / len(values)
    incremental = values[1:] if len(values) > 1 else values
    return {
        "aia_percent": 100.0 * aia,
        "incremental_aia_percent": 100.0
        * sum(incremental)
        / len(incremental),
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0
        * (aia - baseline_aia),
        "aia_delta_vs_sequential_affine_percent_points": 100.0
        * (aia - sequential_aia),
        "final_delta_vs_nme_percent_points": 100.0
        * (values[-1] - baseline[-1]),
        "uses_test_data_for_selection": False,
        "reference_checkpoint": "session_00",
        "recursive_transport": False,
        "training_reused_without_changes": True,
    }


def _aggregate_population_mass_transport(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    affine = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["population_mass_transport"]["accuracy"])
        for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    affine_aia = sum(affine) / len(affine)
    aia = sum(values) / len(values)
    incremental = values[1:] if len(values) > 1 else values
    return {
        "aia_percent": 100.0 * aia,
        "incremental_aia_percent": 100.0
        * sum(incremental)
        / len(incremental),
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0
        * (aia - baseline_aia),
        "aia_delta_vs_uniform_affine_percent_points": 100.0
        * (aia - affine_aia),
        "final_delta_vs_nme_percent_points": 100.0
        * (values[-1] - baseline[-1]),
        "uses_test_data_for_selection": False,
        "support_measure": "persistent_introduction_population_mass",
        "training_reused_without_changes": True,
    }


def _aggregate_moment_calibrated_affine(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    uniform = [float(record["cmpt"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    uniform_aia = sum(uniform) / len(uniform)
    variants: dict[str, Any] = {}
    for mode in CALIBRATION_MODES:
        values = [
            float(record["moment_calibrated_affine"][mode]["accuracy"])
            for record in records
        ]
        aia = sum(values) / len(values)
        incremental = values[1:] if len(values) > 1 else values
        diagnostics = [
            record["moment_calibrated_affine_diagnostics"][mode]
            for record in records[1:]
        ]

        def diagnostic_mean(key: str) -> float:
            return (
                sum(float(item[key]) for item in diagnostics)
                / len(diagnostics)
                if diagnostics
                else 0.0
            )

        variants[mode] = {
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "aia_delta_vs_uniform_affine_percent_points": 100.0
            * (aia - uniform_aia),
            "final_delta_vs_uniform_affine_percent_points": 100.0
            * (values[-1] - uniform[-1]),
            "mean_effective_sample_size": diagnostic_mean(
                "mean_effective_sample_size"
            ),
            "mean_weighted_mean_relative_error": diagnostic_mean(
                "mean_weighted_mean_relative_error"
            ),
            "mean_weighted_second_relative_error": diagnostic_mean(
                "mean_weighted_second_relative_error"
            ),
        }
    return {
        "uniform_affine": {
            "aia_percent": 100.0 * uniform_aia,
            "final_percent": 100.0 * uniform[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (uniform_aia - baseline_aia),
        },
        **variants,
        "uses_test_data_for_selection": False,
        "full_space_second_moment_kernel": True,
        "training_reused_without_changes": True,
    }


def _aggregate_moment_transport(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    affine = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["moment_transport"]["accuracy"])
        for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    affine_aia = sum(affine) / len(affine)
    aia = sum(values) / len(values)
    incremental = values[1:] if len(values) > 1 else values
    reductions = [
        float(record["moment_transport_diagnostics"]["residual_reduction"])
        for record in records[1:]
    ]
    return {
        "aia_percent": 100.0 * aia,
        "incremental_aia_percent": 100.0
        * sum(incremental)
        / len(incremental),
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0
        * (aia - baseline_aia),
        "aia_delta_vs_affine_percent_points": 100.0 * (aia - affine_aia),
        "final_delta_vs_nme_percent_points": 100.0
        * (values[-1] - baseline[-1]),
        "mean_support_residual_reduction": (
            sum(reductions) / len(reductions) if reductions else 0.0
        ),
        "uses_test_data_for_selection": False,
        "stores_introduction_first_and_second_moments": True,
        "training_reused_without_changes": True,
    }


def _aggregate_moment_grid(
    records: Sequence[Mapping[str, Any]],
    specs: Sequence[tuple[str, int, float, float]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    affine = [float(record["cmpt"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    affine_aia = sum(affine) / len(affine)
    candidates: dict[str, dict[str, Any]] = {}
    for key, rank, ridge, scale in specs:
        values = [
            float(record["moment_transport_grid"][key]["accuracy"])
            for record in records
        ]
        validation_distance = [
            float(
                record["moment_transport_grid_validation"][key][
                    "prototype_cosine_distance"
                ]
            )
            for record in records[1:]
        ]
        validation_accuracy = [
            float(
                record["moment_transport_grid_validation"][key][
                    "nme_accuracy"
                ]
            )
            for record in records[1:]
        ]
        validation_margin = [
            float(
                record["moment_transport_grid_validation"][key][
                    "nme_cosine_margin"
                ]
            )
            for record in records[1:]
        ]
        aia = sum(values) / len(values)
        candidates[key] = {
            "rank": int(rank),
            "quadratic_ridge": float(ridge),
            "correction_scale": float(scale),
            "validation_cosine_distance": (
                sum(validation_distance) / len(validation_distance)
                if validation_distance
                else 0.0
            ),
            "validation_nme_accuracy": (
                sum(validation_accuracy) / len(validation_accuracy)
                if validation_accuracy
                else 0.0
            ),
            "validation_nme_cosine_margin": (
                sum(validation_margin) / len(validation_margin)
                if validation_margin
                else 0.0
            ),
            "aia_percent": 100.0 * aia,
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "aia_delta_vs_affine_percent_points": 100.0
            * (aia - affine_aia),
        }
    validation_key = min(
        candidates,
        key=lambda key: (
            -candidates[key]["validation_nme_cosine_margin"],
            -candidates[key]["validation_nme_accuracy"],
            candidates[key]["validation_cosine_distance"],
            key,
        ),
    )
    oracle_key = max(
        candidates,
        key=lambda key: (candidates[key]["aia_percent"], key),
    )
    return {
        "selection_metric": (
            "class-stratified held-out exemplar NME cosine margin"
        ),
        "validation_selected_key": validation_key,
        "validation_selected": candidates[validation_key],
        "test_oracle_key": oracle_key,
        "test_oracle": candidates[oracle_key],
        "test_oracle_is_diagnostic_only": True,
        "candidates": candidates,
    }


def build_old_class_cmpt_means(
    baseline_means: Tensor,
    transported_means: Tensor,
    old_class_count: int,
) -> Tensor:
    """Replace only old rows, keeping current-session NME means unchanged."""

    if baseline_means.ndim != 2 or transported_means.ndim != 2:
        raise ValueError("NME and transported means must be matrices")
    if baseline_means.shape != transported_means.shape:
        raise ValueError("NME and transported means must have one shape")
    boundary = int(old_class_count)
    if not 0 <= boundary <= baseline_means.shape[0]:
        raise ValueError("old_class_count is outside the prototype bank")
    result = baseline_means.detach().clone()
    if boundary > 0:
        result[:boundary] = transported_means[:boundary]
    return result


def build_old_class_interpolated_means(
    baseline_means: Tensor,
    transported_means: Tensor,
    old_class_count: int,
    alpha: float,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Interpolate NME and CMPT means for old classes only.

    ``alpha=0`` is exactly the current-memory NME bank and ``alpha=1`` is
    exactly the existing CMPT bank.  Current-session rows are never mixed.
    """

    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("prototype interpolation alpha must lie in [0, 1]")
    if epsilon <= 0.0:
        raise ValueError("prototype interpolation epsilon must be positive")
    cmpt_means = build_old_class_cmpt_means(
        baseline_means,
        transported_means,
        old_class_count,
    )
    if value == 0.0:
        return baseline_means.detach().clone()
    if value == 1.0:
        return cmpt_means

    boundary = int(old_class_count)
    result = baseline_means.detach().clone()
    if boundary == 0:
        return result
    baseline_old = F.normalize(
        baseline_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    cmpt_old = F.normalize(
        cmpt_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    mixed = (1.0 - value) * baseline_old + value * cmpt_old
    if bool((mixed.norm(dim=1) <= float(epsilon)).any()):
        raise ValueError(
            "prototype interpolation produced a zero-norm old-class row"
        )
    result[:boundary] = F.normalize(mixed, dim=1, eps=epsilon).to(
        dtype=result.dtype, device=result.device
    )
    return result


def build_old_class_adaptive_means(
    baseline_means: Tensor,
    transported_means: Tensor,
    old_class_count: int,
    old_class_alphas: Tensor | Sequence[float],
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Interpolate each old-class row with its own test-independent alpha."""

    boundary = int(old_class_count)
    cmpt_means = build_old_class_cmpt_means(
        baseline_means, transported_means, boundary
    )
    alphas = torch.as_tensor(
        old_class_alphas,
        dtype=torch.float32,
        device=baseline_means.device,
    ).flatten()
    if alphas.numel() != boundary:
        raise ValueError("adaptive alpha count must equal old_class_count")
    if bool((~torch.isfinite(alphas)).any()) or bool(
        ((alphas < 0.0) | (alphas > 1.0)).any()
    ):
        raise ValueError("adaptive class alphas must lie in [0, 1]")
    result = baseline_means.detach().clone()
    if boundary == 0:
        return result
    baseline_old = F.normalize(
        baseline_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    cmpt_old = F.normalize(
        cmpt_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    mixed = (
        (1.0 - alphas[:, None]) * baseline_old
        + alphas[:, None] * cmpt_old
    )
    if bool((mixed.norm(dim=1) <= float(epsilon)).any()):
        raise ValueError("adaptive interpolation produced a zero-norm row")
    result[:boundary] = F.normalize(mixed, dim=1, eps=epsilon).to(
        dtype=result.dtype, device=result.device
    )
    return result


def class_geometric_oracle_alphas(
    baseline_old_means: Tensor,
    transported_old_means: Tensor,
    full_current_old_means: Tensor,
    alpha_grid: Sequence[float],
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, dict[str, float]]:
    """Select per-class alpha by unavailable full-current prototype targets."""

    if not (
        baseline_old_means.ndim == 2
        and baseline_old_means.shape == transported_old_means.shape
        and baseline_old_means.shape == full_current_old_means.shape
    ):
        raise ValueError("geometric-oracle prototype shapes must match")
    grid = torch.as_tensor(
        tuple(float(value) for value in alpha_grid), dtype=torch.float32
    )
    if grid.numel() == 0 or bool((grid < 0.0).any()) or bool(
        (grid > 1.0).any()
    ):
        raise ValueError("geometric-oracle alpha grid must lie in [0, 1]")
    baseline = F.normalize(baseline_old_means.detach().cpu().float(), dim=1)
    transported = F.normalize(
        transported_old_means.detach().cpu().float(), dim=1
    )
    target = F.normalize(
        full_current_old_means.detach().cpu().float(), dim=1
    )
    candidates = (
        (1.0 - grid[:, None, None]) * baseline[None]
        + grid[:, None, None] * transported[None]
    )
    norms = candidates.norm(dim=2)
    normalized = F.normalize(candidates, dim=2, eps=epsilon)
    similarities = (normalized * target[None]).sum(dim=2)
    similarities = similarities.masked_fill(norms <= float(epsilon), -1.0)
    best_indices = similarities.argmax(dim=0)
    class_indices = torch.arange(baseline.shape[0])
    best_alphas = grid[best_indices]
    selected_similarity = similarities[best_indices, class_indices]
    return best_alphas, {
        "baseline_mean_cosine_distance": float(
            (1.0 - (baseline * target).sum(dim=1)).mean().item()
        ),
        "cmpt_mean_cosine_distance": float(
            (1.0 - (transported * target).sum(dim=1)).mean().item()
        ),
        "oracle_mean_cosine_distance": float(
            (1.0 - selected_similarity).mean().item()
        ),
    }


def adaptive_alpha_from_uncertainties(
    memory_uncertainty: Tensor,
    transport_uncertainty: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Return the inverse-risk weight assigned to transported CMPT means."""

    if memory_uncertainty.shape != transport_uncertainty.shape:
        raise ValueError("adaptive uncertainty tensors must have one shape")
    if epsilon <= 0.0:
        raise ValueError("adaptive alpha epsilon must be positive")
    memory = memory_uncertainty.detach().float()
    transport = transport_uncertainty.detach().float()
    if bool((~torch.isfinite(memory)).any()) or bool(
        (~torch.isfinite(transport)).any()
    ):
        raise ValueError("adaptive uncertainties must be finite")
    if bool((memory < 0.0).any()) or bool((transport < 0.0).any()):
        raise ValueError("adaptive uncertainties must be non-negative")
    return (
        (memory + float(epsilon))
        / (memory + transport + 2.0 * float(epsilon))
    ).clamp(0.0, 1.0)


def empirical_bayes_shrink_alphas(
    raw_alphas: Tensor,
    jackknife_variances: Tensor,
    session_alpha: float,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, float]:
    """Shrink noisy class alphas toward their pooled session estimate."""

    raw = raw_alphas.detach().float().flatten()
    variances = jackknife_variances.detach().float().flatten()
    if raw.shape != variances.shape or raw.numel() == 0:
        raise ValueError("raw alpha and jackknife variance shapes must match")
    if bool((variances < 0.0).any()) or bool((~torch.isfinite(variances)).any()):
        raise ValueError("jackknife variances must be finite and non-negative")
    observed = (
        raw.var(unbiased=True)
        if raw.numel() > 1
        else torch.zeros((), dtype=raw.dtype, device=raw.device)
    )
    between_class_variance = (observed - variances.mean()).clamp_min(0.0)
    credibility = between_class_variance / (
        between_class_variance + variances + float(epsilon)
    )
    center = torch.full_like(raw, float(session_alpha))
    shrunken = center + credibility * (raw - center)
    return (
        shrunken.clamp(0.0, 1.0),
        credibility,
        float(between_class_variance.item()),
    )


@dataclass(frozen=True)
class AdaptiveAlphaEstimate:
    session_alpha: float
    raw_class_alphas: Tensor
    shrunken_class_alphas: Tensor
    memory_uncertainties: Tensor
    transport_uncertainties: Tensor
    jackknife_variances: Tensor
    shrinkage_credibilities: Tensor
    between_class_variance: float
    folds: int


def _fit_affine_mapping(
    old_features: Tensor,
    current_features: Tensor,
    *,
    ridge: float,
) -> Tensor:
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    design = torch.cat(
        [old, torch.ones(old.shape[0], 1, dtype=old.dtype)], dim=1
    )
    regularizer = torch.eye(design.shape[1], dtype=design.dtype) * float(ridge)
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ current,
    )


def _apply_affine_mapping(features: Tensor, mapping: Tensor) -> Tensor:
    values = F.normalize(features.detach().float(), dim=1)
    design = torch.cat(
        [values, torch.ones(values.shape[0], 1, dtype=values.dtype)], dim=1
    )
    return F.normalize(design @ mapping, dim=1)


def _mean_estimation_uncertainty(values: Tensor) -> Tensor:
    count = int(values.shape[0])
    if count < 2:
        raise ValueError("mean uncertainty requires at least two exemplars")
    normalized = F.normalize(values.detach().float(), dim=1)
    center = F.normalize(normalized.mean(dim=0, keepdim=True), dim=1)
    return (normalized - center).square().sum() / float(count * (count - 1))


def _transport_estimation_uncertainty(residuals: Tensor) -> Tensor:
    count = int(residuals.shape[0])
    if count < 2:
        raise ValueError("transport uncertainty requires two exemplars")
    mean = residuals.mean(dim=0, keepdim=True)
    squared_bias = mean.square().sum()
    variance_of_mean = (residuals - mean).square().sum() / float(
        count * (count - 1)
    )
    return squared_bias + variance_of_mean


def _cross_fitted_affine_residuals(
    old_features: Tensor,
    current_features: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    folds: int,
    ridge: float,
) -> tuple[Tensor, int]:
    old = F.normalize(old_features.detach().cpu().float(), dim=1)
    current = F.normalize(current_features.detach().cpu().float(), dim=1)
    labels = targets.detach().cpu().long().flatten()
    if old.shape != current.shape or old.shape[0] != labels.numel():
        raise ValueError("cross-fitted affine support shapes do not match")
    expected = set(range(int(num_classes)))
    observed = set(int(value) for value in labels.unique().tolist())
    if observed != expected:
        raise ValueError("adaptive support must cover every contiguous old class")
    minimum_count = min(
        int((labels == class_index).sum().item())
        for class_index in range(int(num_classes))
    )
    effective_folds = min(int(folds), minimum_count)
    if effective_folds < 2:
        raise ValueError("adaptive alpha requires at least two folds")
    fold_ids = torch.empty_like(labels)
    for class_index in range(int(num_classes)):
        positions = torch.where(labels == class_index)[0]
        fold_ids[positions] = torch.arange(positions.numel()) % effective_folds
    residuals = torch.empty_like(current)
    for fold in range(effective_folds):
        held_out = fold_ids == fold
        mapping = _fit_affine_mapping(
            old[~held_out], current[~held_out], ridge=ridge
        )
        predicted = _apply_affine_mapping(old[held_out], mapping)
        residuals[held_out] = current[held_out] - predicted
    return residuals, effective_folds


def estimate_adaptive_alphas(
    support: PairedTransportSupport,
    *,
    num_classes: int,
    folds: int,
    ridge: float,
    epsilon: float = 1.0e-12,
) -> AdaptiveAlphaEstimate:
    """Estimate session, raw class, and EB-shrunken class CMPT weights."""

    residuals, effective_folds = _cross_fitted_affine_residuals(
        support.old_exemplar_features,
        support.current_exemplar_features,
        support.targets,
        num_classes=num_classes,
        folds=folds,
        ridge=ridge,
    )
    labels = support.targets.detach().cpu().long()
    current = support.current_exemplar_features.detach().cpu().float()
    memory_uncertainties: list[Tensor] = []
    transport_uncertainties: list[Tensor] = []
    jackknife_variances: list[Tensor] = []
    for class_index in range(int(num_classes)):
        values = current[labels == class_index]
        class_residuals = residuals[labels == class_index]
        count = int(values.shape[0])
        if count < 3:
            raise ValueError(
                "class-wise adaptive alpha requires at least three exemplars"
            )
        memory_risk = _mean_estimation_uncertainty(values)
        transport_risk = _transport_estimation_uncertainty(class_residuals)
        memory_uncertainties.append(memory_risk)
        transport_uncertainties.append(transport_risk)

        leave_one_out: list[Tensor] = []
        for omitted in range(count):
            keep = torch.arange(count) != omitted
            leave_memory = _mean_estimation_uncertainty(values[keep])
            leave_transport = _transport_estimation_uncertainty(
                class_residuals[keep]
            )
            leave_one_out.append(
                adaptive_alpha_from_uncertainties(
                    leave_memory[None],
                    leave_transport[None],
                    epsilon=epsilon,
                )[0]
            )
        jackknife = torch.stack(leave_one_out)
        jackknife_variances.append(
            float(count - 1)
            / float(count)
            * (jackknife - jackknife.mean()).square().sum()
        )

    memory = torch.stack(memory_uncertainties)
    transport = torch.stack(transport_uncertainties)
    raw = adaptive_alpha_from_uncertainties(
        memory, transport, epsilon=epsilon
    )
    session = float(
        adaptive_alpha_from_uncertainties(
            memory.mean()[None], transport.mean()[None], epsilon=epsilon
        )[0].item()
    )
    shrunken, credibility, between = empirical_bayes_shrink_alphas(
        raw,
        torch.stack(jackknife_variances),
        session,
        epsilon=epsilon,
    )
    return AdaptiveAlphaEstimate(
        session_alpha=session,
        raw_class_alphas=raw,
        shrunken_class_alphas=shrunken,
        memory_uncertainties=memory,
        transport_uncertainties=transport,
        jackknife_variances=torch.stack(jackknife_variances),
        shrinkage_credibilities=credibility,
        between_class_variance=between,
        folds=effective_folds,
    )


def _alpha_key(alpha: float) -> str:
    return format(float(alpha), ".6g")


def _alpha_statistics(values: Tensor) -> dict[str, Any]:
    flattened = values.detach().cpu().float().flatten()
    if flattened.numel() == 0:
        return {
            "alpha_mean": None,
            "alpha_min": None,
            "alpha_max": None,
            "alphas": [],
        }
    return {
        "alpha_mean": float(flattened.mean().item()),
        "alpha_min": float(flattened.min().item()),
        "alpha_max": float(flattened.max().item()),
        "alphas": [float(value) for value in flattened.tolist()],
    }


@torch.inference_mode()
def _evaluate_nme_banks(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    old_class_count: int,
    class_mean_banks: Mapping[str, Tensor],
    *,
    center_strength: float,
    horizontal_flip_query: bool,
) -> dict[str, dict[str, Any]]:
    """Evaluate several prototype banks from one query-feature extraction."""

    if not class_mean_banks:
        raise ValueError("NME bank evaluation requires at least one bank")
    if not 0.0 <= float(center_strength) <= 1.0:
        raise ValueError("NME center_strength must be in [0, 1]")

    collection = collect_features(model, loader, device)
    features = F.normalize(collection.features.to(device).float(), dim=1)
    if horizontal_flip_query:
        flipped = collect_features(
            model, loader, device, horizontal_flip=True
        )
        if not torch.equal(collection.indices, flipped.indices):
            raise RuntimeError("NME query flip rows are misaligned")
        features = F.normalize(
            features
            + F.normalize(flipped.features.to(device).float(), dim=1),
            dim=1,
        )

    targets = collection.targets.to(device).long()
    original_targets = collection.original_targets.long()
    old_mask = targets < int(old_class_count)
    new_mask = ~old_mask
    total_count = int(targets.numel())
    old_count = int(old_mask.sum().item())
    new_count = int(new_mask.sum().item())
    if total_count == 0 or new_count == 0:
        raise ValueError("cannot evaluate an empty NME split")

    results: dict[str, dict[str, Any]] = {}
    expected_shape: tuple[int, int] | None = None
    for name, class_means in class_mean_banks.items():
        means = F.normalize(class_means.to(device).float(), dim=1)
        shape = (int(means.shape[0]), int(means.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all NME prototype banks must have one shape")
        center = means.mean(dim=0, keepdim=True)
        query = features
        if center_strength > 0.0:
            means = F.normalize(
                means - float(center_strength) * center, dim=1
            )
            query = F.normalize(
                features - float(center_strength) * center, dim=1
            )
        predictions = (query @ means.T).argmax(dim=1)
        correct = predictions.eq(targets)
        old_accuracy = (
            None
            if old_count == 0
            else int(correct[old_mask].sum().item()) / old_count
        )
        new_accuracy = int(correct[new_mask].sum().item()) / new_count
        harmonic = None
        if old_accuracy is not None:
            denominator = old_accuracy + new_accuracy
            harmonic = (
                0.0
                if denominator == 0.0
                else 2.0 * old_accuracy * new_accuracy / denominator
            )
        cpu_correct = correct.cpu()
        per_class_accuracy: dict[int, float] = {}
        for class_id in original_targets.unique().tolist():
            mask = original_targets == int(class_id)
            per_class_accuracy[int(class_id)] = int(
                cpu_correct[mask].sum().item()
            ) / int(mask.sum().item())
        results[str(name)] = EvaluationResult(
            accuracy=int(correct.sum().item()) / total_count,
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            harmonic_mean=harmonic,
            per_class_accuracy=per_class_accuracy,
            sample_count=total_count,
        ).to_dict()
    return results


class CMPTCheckpointEvaluator:
    """Evaluate native, current-memory NME, and CMPT on one trajectory."""

    def __init__(
        self,
        settings: CMPTExperimentSettings,
        project_root: str | Path,
        *,
        source_root: str | Path,
        max_sessions: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.project_root = Path(project_root).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        self.progress = progress or (lambda _: None)
        torch.set_num_threads(int(settings.cpu_threads))
        self.checkpoint_paths = discover_checkpoint_paths(
            settings.checkpoint_directory
        )
        self.checkpoints = [
            load_checkpoint(path, map_location="cpu")
            for path in self.checkpoint_paths
        ]
        self.audit = audit_checkpoint_trajectory(self.checkpoints, settings)
        native_specs = [
            resolve_native_classifier(checkpoint)
            for checkpoint in self.checkpoints
        ]
        self.native_classifier = native_specs[0]
        if any(
            spec != self.native_classifier for spec in native_specs[1:]
        ):
            raise ValueError(
                f"{settings.learner}: native classifier contract changes "
                "across sessions"
            )
        if max_sessions is not None:
            count = int(max_sessions)
            if count <= 0:
                raise ValueError("max_sessions must be positive")
            self.checkpoint_paths = self.checkpoint_paths[:count]
            self.checkpoints = self.checkpoints[:count]

    def validation_payload(self) -> dict[str, Any]:
        return {
            "status": "validated",
            "trajectory": self.audit.to_dict(),
            "checkpoint_directory": str(
                self.settings.checkpoint_directory
            ),
            "output_file": str(self.settings.output_file),
            "cmpt": self._cmpt_metadata(),
            "native_classifier": self.native_classifier.to_dict(),
        }

    def _cmpt_metadata(self) -> dict[str, Any]:
        return {
            "transport": self.settings.transport,
            "affine_ridge": (
                self.settings.affine_ridge
                if self.settings.transport == "affine_ridge"
                else None
            ),
            "full_introduction_prototypes": True,
            "evaluation_replacement": "transported_old_classes_only",
            "current_session_prototypes": "current_exemplar_nme",
            "prototype_horizontal_flip": (
                self.settings.prototype_horizontal_flip
            ),
            "support_horizontal_flip": self.settings.support_horizontal_flip,
            "query_horizontal_flip": self.settings.query_horizontal_flip,
            "center_strength": self.settings.center_strength,
            "strict_parity": self.settings.strict_parity,
            "parity_tolerance": self.settings.parity_tolerance,
            "cpu_threads": self.settings.cpu_threads,
            "prototype_interpolation_alphas": list(
                self.settings.prototype_interpolation_alphas
            ),
            "prototype_interpolation_formula": (
                "normalize((1-alpha)*current_memory_nme + "
                "alpha*transported_cmpt), old classes only"
            ),
            "prototype_interpolation_sweep_is_diagnostic": bool(
                self.settings.prototype_interpolation_alphas
            ),
            "adaptive_alpha": {
                "enabled": self.settings.adaptive_alpha_enabled,
                "folds": self.settings.adaptive_alpha_folds,
                "variants": [
                    "session",
                    "raw_class",
                    "shrunken_class",
                ],
                "memory_uncertainty": (
                    "directional exemplar-mean variance"
                ),
                "transport_uncertainty": (
                    "stratified out-of-fold affine residual bias plus "
                    "variance-of-mean"
                ),
                "class_shrinkage": (
                    "jackknife empirical-Bayes shrinkage toward session alpha"
                ),
                "uses_test_data": False,
            },
            "oracle_diagnostics": {
                "accuracy_enabled": self.settings.accuracy_oracle_enabled,
                "class_geometric_enabled": (
                    self.settings.class_geometric_oracle_enabled
                ),
                "full_mean_enabled": (
                    self.settings.full_mean_oracle_enabled
                ),
                "alpha_grid": list(self.settings.oracle_alpha_grid),
                "oracle_only": bool(
                    self.settings.accuracy_oracle_enabled
                    or self.settings.class_geometric_oracle_enabled
                    or self.settings.full_mean_oracle_enabled
                ),
                "accuracy_oracle_uses_test_labels": (
                    self.settings.accuracy_oracle_enabled
                ),
                "class_geometric_uses_full_old_training_data": (
                    self.settings.class_geometric_oracle_enabled
                ),
                "full_mean_uses_all_seen_training_data": (
                    self.settings.full_mean_oracle_enabled
                ),
            },
            "component_ablation": {
                "enabled": self.settings.component_ablation_enabled,
                "class_translation_formula": (
                    "p_t = normalize(p_previous + "
                    "mean_current_exemplar - mean_old_exemplar)"
                ),
                "combined_formula": (
                    "p_t = normalize(T(p_previous) + "
                    "mean_current_exemplar - T(mean_old_exemplar))"
                ),
                "paired_old_exemplars_only": True,
                "uses_test_data": False,
            },
            "neighbor_affine": {
                "enabled": self.settings.neighbor_affine_enabled,
                "classes_per_neighborhood": (
                    self.settings.neighbor_affine_classes
                ),
                "selection_space": "previous-model exemplar class means",
                "similarity": "cosine",
                "overlapping_neighborhoods": True,
                "includes_center_class": True,
                "uses_test_data": False,
            },
            "herding_extrapolation": {
                "enabled": self.settings.herding_extrapolation_enabled,
                "full_prefix": self.settings.herding_full_prefix,
                "reference_prefix": self.settings.herding_reference_prefix,
                "formula": (
                    "normalize((K*m_K-J*m_J)/(K-J))"
                ),
                "preserves_stored_herding_priority_order": True,
                "uses_test_data": False,
            },
            "persistent_quadrature": {
                "enabled": self.settings.persistent_quadrature_enabled,
                "uniform_ridge": (
                    self.settings.persistent_quadrature_ridge
                ),
                "max_iterations": (
                    self.settings.persistent_quadrature_iterations
                ),
                "multiview": self.settings.persistent_quadrature_multiview,
                "previous_model_view": (
                    self.settings.persistent_quadrature_previous_model_view
                ),
                "fit_time": "class_introduction_only",
                "fit_target": "full_training_population_mean",
                "future_inputs": "same_retained_exemplar_features",
                "simplex_weights": True,
                "replace_old_classes_only": True,
                "uses_test_data": False,
            },
            "canonical_reference": {
                "enabled": self.settings.canonical_reference_enabled,
                "reference_checkpoint": "session_00",
                "affine_ridge": self.settings.canonical_reference_ridge,
                "source_features": (
                    "frozen_session_00_features_for_every_arriving_class"
                ),
                "fit_support": "retained_old_exemplar_identities",
                "direct_reference_to_current_mapping": True,
                "recursive_transport": False,
                "replace_old_classes_only": True,
                "uses_test_data": False,
            },
            "population_mass_transport": {
                "enabled": self.settings.population_mass_transport_enabled,
                "affine_ridge": (
                    self.settings.population_mass_transport_ridge
                ),
                "support_weights": (
                    "persistent_introduction_time_quadrature_mass"
                ),
                "same_exemplar_identity_across_sessions": True,
                "weighted_affine_objective": True,
                "replace_old_classes_only": True,
                "uses_test_data": False,
            },
            "moment_calibrated_affine": {
                "enabled": self.settings.moment_calibrated_affine_enabled,
                "uniform_ridge": (
                    self.settings.moment_calibrated_affine_uniform_ridge
                ),
                "max_iterations": (
                    self.settings.moment_calibrated_affine_iterations
                ),
                "affine_ridge": (
                    self.settings.moment_calibrated_affine_ridge
                ),
                "variants": list(CALIBRATION_MODES),
                "stored_statistics": (
                    "full-population first and second moments at class "
                    "introduction"
                ),
                "weight_fit_time": "every incremental transition",
                "second_moment_estimator": (
                    "exact full-space centered degree-2 kernel"
                ),
                "simplex_weights": True,
                "replace_old_classes_only": True,
                "uses_test_data": False,
            },
            "moment_transport": {
                "enabled": self.settings.moment_transport_enabled,
                "mode": self.settings.moment_transport_mode,
                "rank": self.settings.moment_transport_rank,
                "affine_ridge": (
                    self.settings.moment_transport_affine_ridge
                ),
                "quadratic_ridge": (
                    self.settings.moment_transport_quadratic_ridge
                ),
                "residual_covariance_scale": (
                    self.settings.moment_transport_residual_covariance_scale
                ),
                "stored_statistics": (
                    "full-population first and second moments at class "
                    "introduction"
                ),
                "transition_model": (
                    "global affine plus "
                    + (
                        "full-space homogeneous degree-2 kernel residual"
                        if self.settings.moment_transport_mode
                        == "polynomial_kernel"
                        else "low-rank quadratic conditional mean"
                    )
                ),
                "fit_support": "paired retained old exemplars",
                "replace_old_classes_only": True,
                "uses_test_data": False,
                "grid": {
                    "enabled": self.settings.moment_grid_enabled,
                    "ranks": list(self.settings.moment_grid_ranks),
                    "quadratic_ridges": list(
                        self.settings.moment_grid_quadratic_ridges
                    ),
                    "correction_scales": list(
                        self.settings.moment_grid_correction_scales
                    ),
                    "validation_folds": (
                        self.settings.moment_grid_validation_folds
                    ),
                    "validation_fold": (
                        self.settings.moment_grid_validation_fold
                    ),
                    "selection_metric": (
                        "held-out old-exemplar NME cosine margin"
                    ),
                    "selection_uses_test_data": False,
                    "test_oracle_is_diagnostic_only": True,
                },
            },
            "training_reused_without_changes": True,
        }

    def _trainer(self) -> UnifiedTable1Trainer:
        config = copy.deepcopy(self.checkpoints[0]["config"])
        config["device"] = self.settings.device
        config["output"] = {
            "directory": str(
                self.project_root / "outputs" / "cmpt" / "_runtime"
            ),
            "run_name": re.sub(
                r"[^a-zA-Z0-9_.-]+", "_", self.settings.learner.lower()
            ),
        }
        return UnifiedTable1Trainer(
            config,
            self.project_root,
            max_sessions=len(self.checkpoints),
        )

    def _partial_payload(
        self,
        records: list[dict[str, Any]],
        *,
        status: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 2,
            "status": status,
            "learner": self.settings.learner,
            "trajectory": self.audit.to_dict(),
            "native_classifier": self.native_classifier.to_dict(),
            "checkpoint_directory": str(
                self.settings.checkpoint_directory
            ),
            "evaluated_checkpoint_count": len(self.checkpoints),
            "cmpt": self._cmpt_metadata(),
            "cil_valid_data_access": not (
                self.settings.class_geometric_oracle_enabled
                or self.settings.full_mean_oracle_enabled
            ),
            "test_labels_used_for_selection": (
                self.settings.accuracy_oracle_enabled
            ),
            "oracle_only": bool(
                self.settings.accuracy_oracle_enabled
                or self.settings.class_geometric_oracle_enabled
                or self.settings.full_mean_oracle_enabled
            ),
            "checkpoint_weights_modified": False,
            "elapsed_seconds": elapsed_seconds,
            "git_commit": git_commit(self.project_root),
            "source_provenance": build_exploration_provenance(
                self.source_root, self.project_root / "src_explore"
            ),
            "records": records,
        }
        if records:
            payload["summary"] = _aggregate(records)
            if self.settings.prototype_interpolation_alphas:
                payload["summary"]["prototype_interpolation"] = (
                    _aggregate_interpolation(
                        records,
                        self.settings.prototype_interpolation_alphas,
                    )
                )
            if self.settings.adaptive_alpha_enabled:
                payload["summary"]["adaptive_alpha"] = (
                    _aggregate_adaptive_alpha(records)
                )
            if self.settings.accuracy_oracle_enabled:
                payload["summary"]["accuracy_oracle"] = (
                    _aggregate_accuracy_oracle(
                        records, self.settings.oracle_alpha_grid
                    )
                )
            if self.settings.class_geometric_oracle_enabled:
                payload["summary"]["class_geometric_oracle"] = (
                    _aggregate_class_geometric_oracle(records)
                )
            if self.settings.full_mean_oracle_enabled:
                payload["summary"]["full_mean_oracle"] = (
                    _aggregate_full_mean_oracle(records)
                )
            if self.settings.component_ablation_enabled:
                payload["summary"]["component_ablation"] = (
                    _aggregate_component_ablation(records)
                )
            if self.settings.neighbor_affine_enabled:
                payload["summary"]["neighbor_affine"] = (
                    _aggregate_neighbor_affine(records)
                )
            if self.settings.herding_extrapolation_enabled:
                payload["summary"]["herding_extrapolation"] = (
                    _aggregate_herding_extrapolation(records)
                )
            if self.settings.persistent_quadrature_enabled:
                payload["summary"]["persistent_quadrature"] = (
                    _aggregate_persistent_quadrature(records)
                )
            if self.settings.canonical_reference_enabled:
                payload["summary"]["canonical_reference"] = (
                    _aggregate_canonical_reference(records)
                )
            if self.settings.population_mass_transport_enabled:
                payload["summary"]["population_mass_transport"] = (
                    _aggregate_population_mass_transport(records)
                )
            if self.settings.moment_calibrated_affine_enabled:
                payload["summary"]["moment_calibrated_affine"] = (
                    _aggregate_moment_calibrated_affine(records)
                )
            if self.settings.moment_transport_enabled:
                payload["summary"]["moment_transport"] = (
                    _aggregate_moment_transport(records)
                )
            if self.settings.moment_grid_enabled:
                payload["summary"]["moment_transport_grid"] = (
                    _aggregate_moment_grid(
                        records,
                        _moment_grid_specs(
                            self.settings.moment_grid_ranks,
                            self.settings.moment_grid_quadratic_ridges,
                            self.settings.moment_grid_correction_scales,
                        ),
                    )
                )
        return payload

    def _evaluate_native(
        self,
        model: nn.Module,
        loader: Any,
        device: torch.device,
        old_class_count: int,
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.native_classifier.distinct_from_nme:
            return copy.deepcopy(dict(baseline))
        view = _NativeClassifierView(model, self.native_classifier)
        return evaluate_classifier(
            view,
            loader,
            device,
            old_class_count,
        ).to_dict()

    def _upgrade_existing_native_output(
        self,
        output: Path,
    ) -> dict[str, Any]:
        """Add native metrics to a completed schema-v1 result atomically."""

        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise FileExistsError(
                f"CMPT output exists but is not complete: {output}; "
                "pass --force to replace it"
            )
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != len(
            self.checkpoints
        ):
            raise ValueError(
                "existing CMPT result does not match the checkpoint count"
            )
        if any("native" in record for record in records):
            raise FileExistsError(
                f"CMPT output already contains native results: {output}; "
                "pass --force to replace it"
            )

        trainer = (
            self._trainer()
            if self.native_classifier.distinct_from_nme
            else None
        )
        started = time.perf_counter()
        for record, checkpoint in zip(records, self.checkpoints):
            session_id = int(checkpoint["session_id"])
            if int(record.get("session_id", -1)) != session_id:
                raise ValueError(
                    "existing CMPT record order does not match checkpoints"
                )
            baseline = record.get("baseline")
            if not isinstance(baseline, Mapping):
                raise ValueError("existing CMPT record lacks its NME result")
            if self.native_classifier.distinct_from_nme:
                assert trainer is not None
                old_class_count = trainer.protocol.session(session_id).start
                model = _load_model(trainer, checkpoint, session_id)
                test_dataset = trainer.data.cumulative_test_dataset(session_id)
                test_loader = trainer._loader(
                    test_dataset,
                    shuffle=False,
                    session_id=session_id + 11000,
                )
                native = self._evaluate_native(
                    model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    baseline,
                )
                del model
            else:
                native = copy.deepcopy(dict(baseline))
            record["native"] = native
            self.progress(
                f"{self.settings.learner} S{session_id}: "
                f"Native={100.0 * float(native['accuracy']):.3f} "
                "(existing NME/CMPT preserved)"
            )

        payload["schema_version"] = 2
        payload["native_classifier"] = self.native_classifier.to_dict()
        payload["summary"] = _aggregate(records)
        payload["native_evaluation"] = {
            "added_to_existing_result": True,
            "training_reused_without_changes": True,
            "checkpoint_weights_modified": False,
            "test_labels_used_for_selection": False,
            "elapsed_seconds": time.perf_counter() - started,
            "git_commit": git_commit(self.project_root),
            "source_provenance": build_exploration_provenance(
                self.source_root, self.project_root / "src_explore"
            ),
        }
        temporary = output.with_suffix(output.suffix + ".native-upgrade.tmp")
        dump_json(payload, temporary)
        temporary.replace(output)
        return payload

    def run(
        self,
        *,
        output_file: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        output = (
            self.settings.output_file
            if output_file is None
            else Path(output_file).expanduser().resolve()
        )
        if output.exists() and not force:
            return self._upgrade_existing_native_output(output)
        trainer = self._trainer()
        records: list[dict[str, Any]] = []
        canonical_reference_model = (
            _load_model(trainer, self.checkpoints[0], 0)
            if self.settings.canonical_reference_enabled
            else None
        )
        canonical_reference_bank: Tensor | None = None
        transported: Tensor | None = None
        population_mass_transport_bank: Tensor | None = None
        moment_transport_means: Tensor | None = None
        moment_transport_seconds: Tensor | None = None
        moment_grid_specs = _moment_grid_specs(
            self.settings.moment_grid_ranks,
            self.settings.moment_grid_quadratic_ridges,
            self.settings.moment_grid_correction_scales,
        )
        moment_grid_states: dict[str, MomentGridState] = {}
        moment_calibrated_states: dict[
            str, MomentCalibratedAffineState
        ] = {}
        class_translation_bank: Tensor | None = None
        combined_bank: Tensor | None = None
        neighbor_affine_bank: Tensor | None = None
        persistent_quadrature_weights: list[Tensor] = []
        previous_checkpoint: Mapping[str, Any] | None = None
        started = time.perf_counter()

        for checkpoint_path, checkpoint in zip(
            self.checkpoint_paths, self.checkpoints
        ):
            session_started = time.perf_counter()
            session_id = int(checkpoint["session_id"])
            seen = trainer.protocol.session(session_id).stop
            old_class_count = trainer.protocol.session(session_id).start
            current_model = _load_model(trainer, checkpoint, session_id)
            trainer.model = current_model
            trainer.memory = ExemplarMemory.from_state_dict(
                checkpoint["memory"]
            )
            paired_support: PairedTransportSupport | None = None
            quadrature_previous_model: nn.Module | None = None
            moment_direct_bank: Tensor | None = None
            moment_grid_direct_banks: dict[str, Tensor] = {}
            moment_grid_validation: dict[str, dict[str, float]] = {}
            moment_calibrated_diagnostics: dict[str, dict[str, Any]] = {}
            introduction_prototypes: Tensor | None = None
            introduction_moment_means: Tensor | None = None
            introduction_moment_seconds: Tensor | None = None
            introduction_population_count: int | None = None
            if (
                self.settings.moment_transport_enabled
                or self.settings.moment_calibrated_affine_enabled
            ):
                (
                    introduction_prototypes,
                    introduction_moment_means,
                    introduction_moment_seconds,
                    introduction_population_count,
                ) = _full_introduction_statistics(
                    trainer,
                    current_model,
                    session_id,
                    horizontal_flip=(
                        self.settings.prototype_horizontal_flip
                    ),
                )

            if session_id == 0:
                transported = (
                    introduction_prototypes
                    if introduction_prototypes is not None
                    else _full_introduction_prototypes(
                        trainer,
                        current_model,
                        session_id,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                    )
                )
                transport_diagnostics = {
                    "initialized": True,
                    "support_count": 0,
                    "fit_residual": None,
                }
                if self.settings.population_mass_transport_enabled:
                    population_mass_transport_bank = (
                        transported.detach().clone()
                    )
                    population_mass_transport_diagnostics = {
                        "initialized": True,
                        "support_count": 0,
                        "fit_residual": None,
                        "mean_support_mass": None,
                        "support_mass_coefficient_of_variation": None,
                    }
                if self.settings.moment_transport_enabled:
                    if (
                        introduction_moment_means is None
                        or introduction_moment_seconds is None
                        or introduction_population_count is None
                    ):
                        raise RuntimeError(
                            "shared introduction moments are missing"
                        )
                    moment_transport_means = introduction_moment_means
                    moment_transport_seconds = introduction_moment_seconds
                    moment_population_count = (
                        introduction_population_count
                    )
                    moment_direct_bank = F.normalize(
                        moment_transport_means, dim=1
                    )
                    moment_transport_diagnostics = {
                        "initialized": True,
                        "mode": self.settings.moment_transport_mode,
                        "rank": self.settings.moment_transport_rank,
                        "population_count": moment_population_count,
                        "support_count": 0,
                        "affine_fit_residual": None,
                        "quadratic_fit_residual": None,
                        "residual_reduction": 0.0,
                    }
                    if self.settings.moment_grid_enabled:
                        for key, _, _, _ in moment_grid_specs:
                            moment_grid_states[key] = MomentGridState(
                                means=moment_transport_means.detach().clone(),
                                second_moments=(
                                    moment_transport_seconds.detach().clone()
                                ),
                                direct_bank=moment_direct_bank.detach().clone(),
                            )
                            moment_grid_direct_banks[key] = (
                                moment_direct_bank.detach().clone()
                            )
                            moment_grid_validation[key] = {
                                "prototype_cosine_distance": 0.0,
                                "nme_accuracy": 1.0,
                                "nme_cosine_margin": 0.0,
                            }
                if self.settings.moment_calibrated_affine_enabled:
                    if (
                        introduction_prototypes is None
                        or introduction_moment_means is None
                        or introduction_moment_seconds is None
                    ):
                        raise RuntimeError(
                            "moment-calibrated affine initialization lacks "
                            "full-population statistics"
                        )
                    for mode in CALIBRATION_MODES:
                        moment_calibrated_states[mode] = (
                            MomentCalibratedAffineState(
                                prototypes=(
                                    introduction_prototypes.detach().clone()
                                ),
                                means=(
                                    introduction_moment_means.detach().clone()
                                ),
                                second_moments=(
                                    introduction_moment_seconds.detach().clone()
                                ),
                            )
                        )
                        moment_calibrated_diagnostics[mode] = {
                            "initialized": True,
                            "mode": mode,
                            "support_count": 0,
                            "fit_residual": None,
                            "mean_effective_sample_size": float(
                                self.settings.expected_exemplars_per_class
                            ),
                            "mean_weighted_mean_relative_error": 0.0,
                            "mean_weighted_second_relative_error": 0.0,
                        }
                if self.settings.component_ablation_enabled:
                    class_translation_bank = transported.detach().clone()
                    combined_bank = transported.detach().clone()
                    component_diagnostics = {
                        "initialized": True,
                        "class_translation_mean_norm": None,
                        "class_translation_max_norm": None,
                        "class_residual_mean_norm": None,
                        "class_residual_max_norm": None,
                    }
                if self.settings.neighbor_affine_enabled:
                    neighbor_affine_bank = transported.detach().clone()
                    neighbor_affine_diagnostics = {
                        "initialized": True,
                        "classes_per_neighborhood": (
                            self.settings.neighbor_affine_classes
                        ),
                        "fit_residual_mean": None,
                        "fit_residual_max": None,
                    }
            else:
                if previous_checkpoint is None or transported is None:
                    raise RuntimeError("CMPT transition lacks previous state")
                previous_model = _load_model(
                    trainer, previous_checkpoint, session_id - 1
                )
                quadrature_previous_model = previous_model
                trainer.memory = ExemplarMemory.from_state_dict(
                    previous_checkpoint["memory"]
                )
                paired_support = _paired_support_features(
                    trainer,
                    previous_model,
                    current_model,
                    session_id - 1,
                    horizontal_flip=(
                        self.settings.support_horizontal_flip
                    ),
                )
                if self.settings.moment_calibrated_affine_enabled:
                    if (
                        introduction_prototypes is None
                        or introduction_moment_means is None
                        or introduction_moment_seconds is None
                    ):
                        raise RuntimeError(
                            "moment-calibrated affine transition lacks new "
                            "full-population statistics"
                        )
                    (
                        calibration_old_support,
                        _,
                        calibration_support_count,
                    ) = _moment_support_from_paired(paired_support)
                    updated_calibrated_states: dict[
                        str, MomentCalibratedAffineState
                    ] = {}
                    for mode in CALIBRATION_MODES:
                        state = moment_calibrated_states.get(mode)
                        if state is None or state.prototypes.shape[0] != (
                            old_class_count
                        ):
                            raise RuntimeError(
                                f"moment-calibrated affine {mode} state is "
                                "missing old classes"
                            )
                        exemplar_weights, weight_diagnostics = (
                            fit_moment_calibration_weights(
                                calibration_old_support,
                                paired_support.targets,
                                state.means,
                                state.second_moments,
                                mode=mode,
                                uniform_ridge=(
                                    self.settings.moment_calibrated_affine_uniform_ridge
                                ),
                                max_iterations=(
                                    self.settings.moment_calibrated_affine_iterations
                                ),
                            )
                        )
                        fit_weights = expand_exemplar_weights(
                            exemplar_weights,
                            paired_support.fit_support_count,
                        )
                        (
                            calibrated_old_prototypes,
                            calibrated_mapping,
                            calibrated_residual,
                        ) = population_weighted_affine_transport(
                            state.prototypes,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                            fit_weights,
                            ridge=(
                                self.settings.moment_calibrated_affine_ridge
                            ),
                        )
                        calibrated_old_means, calibrated_old_seconds = (
                            transport_first_second_moments_affine(
                                state.means,
                                state.second_moments,
                                calibrated_mapping,
                            )
                        )
                        updated_calibrated_states[mode] = (
                            MomentCalibratedAffineState(
                                prototypes=torch.cat(
                                    [
                                        calibrated_old_prototypes.cpu(),
                                        introduction_prototypes.cpu(),
                                    ],
                                    dim=0,
                                ),
                                means=torch.cat(
                                    [
                                        calibrated_old_means.cpu(),
                                        introduction_moment_means.cpu(),
                                    ],
                                    dim=0,
                                ),
                                second_moments=torch.cat(
                                    [
                                        calibrated_old_seconds.cpu(),
                                        introduction_moment_seconds.cpu(),
                                    ],
                                    dim=0,
                                ),
                            )
                        )
                        moment_calibrated_diagnostics[mode] = {
                            "initialized": False,
                            "fit_residual": calibrated_residual,
                            "affine_ridge": (
                                self.settings.moment_calibrated_affine_ridge
                            ),
                            "moment_support_count": (
                                calibration_support_count
                            ),
                            **weight_diagnostics,
                        }
                    moment_calibrated_states = updated_calibrated_states
                if self.settings.population_mass_transport_enabled:
                    if population_mass_transport_bank is None:
                        raise RuntimeError(
                            "population-mass transport lacks prototype bank"
                        )
                    if len(persistent_quadrature_weights) != old_class_count:
                        raise RuntimeError(
                            "population-mass transport lacks old class weights"
                        )
                    old_weight_bank = torch.stack(
                        persistent_quadrature_weights
                    )
                    support_masses = _quadrature_fit_support_weights(
                        old_weight_bank,
                        paired_support.fit_support_count,
                    )
                    (
                        population_mass_old,
                        population_mass_mapping,
                        population_mass_residual,
                    ) = population_weighted_affine_transport(
                        population_mass_transport_bank,
                        paired_support.old_fit_features,
                        paired_support.current_fit_features,
                        support_masses,
                        ridge=(
                            self.settings.population_mass_transport_ridge
                        ),
                    )
                    normalized_masses = support_masses / support_masses.mean()
                    mass_linear = population_mass_mapping[:-1]
                    mass_bias = population_mass_mapping[-1]
                    mass_identity = torch.eye(
                        mass_linear.shape[0],
                        dtype=mass_linear.dtype,
                        device=mass_linear.device,
                    )
                    population_mass_transport_diagnostics = {
                        "initialized": False,
                        "support_count": paired_support.fit_support_count,
                        "fit_residual": population_mass_residual,
                        "affine_ridge": (
                            self.settings.population_mass_transport_ridge
                        ),
                        "mean_support_mass": float(
                            normalized_masses.mean().item()
                        ),
                        "support_mass_coefficient_of_variation": float(
                            normalized_masses.std(unbiased=False).item()
                        ),
                        "linear_identity_deviation": float(
                            (mass_linear - mass_identity).norm().item()
                            / mass_linear.shape[0] ** 0.5
                        ),
                        "bias_norm": float(mass_bias.norm().item()),
                    }
                if self.settings.moment_transport_enabled:
                    if (
                        moment_transport_means is None
                        or moment_transport_seconds is None
                    ):
                        raise RuntimeError(
                            "moment transport lacks prior distribution state"
                        )
                    (
                        moment_old_support,
                        moment_current_support,
                        moment_support_count,
                    ) = _moment_support_from_paired(paired_support)
                    if (
                        self.settings.moment_transport_mode
                        == "polynomial_kernel"
                    ):
                        fit_device = torch.device(self.settings.device)
                        moment_mapping = (
                            fit_polynomial_kernel_moment_transport(
                                moment_old_support.to(fit_device),
                                moment_current_support.to(fit_device),
                                affine_ridge=(
                                    self.settings.moment_transport_affine_ridge
                                ),
                                quadratic_ridge=(
                                    self.settings.moment_transport_quadratic_ridge
                                ),
                            )
                        )
                        (
                            moment_old_means,
                            moment_old_seconds,
                            moment_old_prototypes,
                        ) = apply_polynomial_kernel_moment_transport(
                            moment_transport_means,
                            moment_transport_seconds,
                            moment_mapping,
                            residual_covariance_scale=(
                                self.settings.moment_transport_residual_covariance_scale
                            ),
                        )
                    else:
                        moment_mapping = fit_low_rank_moment_transport(
                            moment_old_support,
                            moment_current_support,
                            rank=self.settings.moment_transport_rank,
                            affine_ridge=(
                                self.settings.moment_transport_affine_ridge
                            ),
                            quadratic_ridge=(
                                self.settings.moment_transport_quadratic_ridge
                            ),
                        )
                        (
                            moment_old_means,
                            moment_old_seconds,
                            moment_old_prototypes,
                        ) = apply_low_rank_moment_transport(
                            moment_transport_means,
                            moment_transport_seconds,
                            moment_mapping,
                            residual_covariance_scale=(
                                self.settings.moment_transport_residual_covariance_scale
                            ),
                        )
                    (
                        moment_new_means,
                        moment_new_seconds,
                    ) = (
                        introduction_moment_means,
                        introduction_moment_seconds,
                    )
                    if (
                        moment_new_means is None
                        or moment_new_seconds is None
                        or introduction_population_count is None
                    ):
                        raise RuntimeError(
                            "shared new-class moments are missing"
                        )
                    moment_population_count = introduction_population_count
                    moment_transport_means = torch.cat(
                        [moment_old_means.cpu(), moment_new_means.cpu()], dim=0
                    )
                    moment_transport_seconds = torch.cat(
                        [moment_old_seconds.cpu(), moment_new_seconds.cpu()],
                        dim=0,
                    )
                    moment_direct_bank = torch.cat(
                        [
                            moment_old_prototypes.cpu(),
                            F.normalize(moment_new_means.float(), dim=1).cpu(),
                        ],
                        dim=0,
                    )
                    affine_fit = moment_mapping.affine_fit_residual
                    quadratic_fit = moment_mapping.quadratic_fit_residual
                    moment_transport_diagnostics = {
                        "initialized": False,
                        "mode": self.settings.moment_transport_mode,
                        "rank": getattr(moment_mapping, "rank", None),
                        "population_count": moment_population_count,
                        "support_count": moment_support_count,
                        "affine_fit_residual": affine_fit,
                        "quadratic_fit_residual": quadratic_fit,
                        "residual_reduction": (
                            (affine_fit - quadratic_fit)
                            / max(affine_fit, 1.0e-12)
                        ),
                        "affine_ridge": (
                            self.settings.moment_transport_affine_ridge
                        ),
                        "quadratic_ridge": (
                            self.settings.moment_transport_quadratic_ridge
                        ),
                    }
                    if self.settings.moment_grid_enabled:
                        if (
                            introduction_moment_means is None
                            or introduction_moment_seconds is None
                        ):
                            raise RuntimeError(
                                "moment grid lacks new-class moments"
                            )
                        if len(moment_grid_states) != len(moment_grid_specs):
                            raise RuntimeError(
                                "moment grid lacks prior candidate states"
                            )
                        grid_mappings = fit_low_rank_moment_transport_grid(
                            moment_old_support,
                            moment_current_support,
                            ranks=self.settings.moment_grid_ranks,
                            quadratic_ridges=(
                                self.settings.moment_grid_quadratic_ridges
                            ),
                            affine_ridge=(
                                self.settings.moment_transport_affine_ridge
                            ),
                        )
                        fit_indices, validation_indices = (
                            _stratified_moment_grid_split(
                                paired_support.targets,
                                folds=(
                                    self.settings.moment_grid_validation_folds
                                ),
                                validation_fold=(
                                    self.settings.moment_grid_validation_fold
                                ),
                            )
                        )
                        validation_mappings = (
                            fit_low_rank_moment_transport_grid(
                                moment_old_support[fit_indices],
                                moment_current_support[fit_indices],
                                ranks=self.settings.moment_grid_ranks,
                                quadratic_ridges=(
                                    self.settings.moment_grid_quadratic_ridges
                                ),
                                affine_ridge=(
                                    self.settings.moment_transport_affine_ridge
                                ),
                            )
                        )
                        validation_features = moment_current_support[
                            validation_indices
                        ]
                        validation_targets = paired_support.targets[
                            validation_indices
                        ]
                        updated_grid_states: dict[str, MomentGridState] = {}
                        for key, rank, ridge, scale in moment_grid_specs:
                            prior_state = moment_grid_states[key]
                            mapping_key = (rank, ridge)
                            (
                                grid_old_means,
                                grid_old_seconds,
                                grid_old_prototypes,
                            ) = apply_low_rank_moment_transport(
                                prior_state.means,
                                prior_state.second_moments,
                                grid_mappings[mapping_key],
                                residual_covariance_scale=0.0,
                                correction_scale=scale,
                            )
                            grid_means = torch.cat(
                                [
                                    grid_old_means.cpu(),
                                    introduction_moment_means.cpu(),
                                ],
                                dim=0,
                            )
                            grid_seconds = torch.cat(
                                [
                                    grid_old_seconds.cpu(),
                                    introduction_moment_seconds.cpu(),
                                ],
                                dim=0,
                            )
                            grid_direct = torch.cat(
                                [
                                    grid_old_prototypes.cpu(),
                                    F.normalize(
                                        introduction_moment_means.float(),
                                        dim=1,
                                    ).cpu(),
                                ],
                                dim=0,
                            )
                            updated_grid_states[key] = MomentGridState(
                                means=grid_means,
                                second_moments=grid_seconds,
                                direct_bank=grid_direct,
                            )
                            moment_grid_direct_banks[key] = grid_direct
                            _, _, validation_prototypes = (
                                apply_low_rank_moment_transport(
                                    prior_state.means,
                                    prior_state.second_moments,
                                    validation_mappings[mapping_key],
                                    residual_covariance_scale=0.0,
                                    correction_scale=scale,
                                )
                            )
                            moment_grid_validation[key] = (
                                _moment_grid_validation_error(
                                    validation_prototypes,
                                    validation_features,
                                    validation_targets,
                                )
                            )
                        moment_grid_states = updated_grid_states
                if self.settings.transport == "rigid_procrustes":
                    transported_old, rotation, translation, residual = (
                        rigid_procrustes_transport(
                            transported,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                        )
                    )
                    transport_diagnostics = {
                        "initialized": False,
                        "support_count": paired_support.fit_support_count,
                        "fit_residual": residual,
                        "rotation_orthogonality_error": float(
                            (
                                rotation.T @ rotation
                                - torch.eye(rotation.shape[0])
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                        "translation_norm": float(translation.norm().item()),
                    }
                else:
                    transported_old, mapping, residual = affine_ridge_transport(
                        transported,
                        paired_support.old_fit_features,
                        paired_support.current_fit_features,
                        ridge=self.settings.affine_ridge,
                    )
                    linear = mapping[:-1]
                    bias = mapping[-1]
                    identity = torch.eye(
                        linear.shape[0],
                        device=linear.device,
                        dtype=linear.dtype,
                    )
                    transport_diagnostics = {
                        "initialized": False,
                        "support_count": paired_support.fit_support_count,
                        "fit_residual": residual,
                        "affine_ridge": self.settings.affine_ridge,
                        "linear_identity_deviation": float(
                            (linear - identity).norm().item()
                            / linear.shape[0] ** 0.5
                        ),
                        "bias_norm": float(bias.norm().item()),
                    }
                if self.settings.neighbor_affine_enabled:
                    if neighbor_affine_bank is None:
                        raise RuntimeError(
                            "neighbor-affine transition lacks prototype bank"
                        )
                    (
                        neighbor_affine_old,
                        neighborhoods,
                        neighbor_residuals,
                    ) = local_neighbor_affine_transport(
                        neighbor_affine_bank,
                        paired_support.old_fit_features,
                        paired_support.current_fit_features,
                        paired_support.fit_targets,
                        classes_per_neighborhood=(
                            self.settings.neighbor_affine_classes
                        ),
                        ridge=self.settings.affine_ridge,
                    )
                    neighbor_affine_diagnostics = {
                        "initialized": False,
                        "classes_per_neighborhood": (
                            self.settings.neighbor_affine_classes
                        ),
                        "fit_residual_mean": float(
                            neighbor_residuals.mean().item()
                        ),
                        "fit_residual_max": float(
                            neighbor_residuals.max().item()
                        ),
                        "neighborhoods": neighborhoods.tolist(),
                    }
                if self.settings.component_ablation_enabled:
                    if (
                        class_translation_bank is None
                        or combined_bank is None
                    ):
                        raise RuntimeError(
                            "component ablation transition lacks prototype banks"
                        )
                    if self.settings.transport != "affine_ridge":
                        raise RuntimeError(
                            "component ablation transition lacks affine mapping"
                        )
                    class_translation_old, class_translations = (
                        classwise_translation_transport(
                            class_translation_bank,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                            paired_support.fit_targets,
                        )
                    )
                    combined_old, class_residuals = (
                        affine_class_residual_transport(
                            combined_bank,
                            mapping,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                            paired_support.fit_targets,
                        )
                    )
                    translation_norms = class_translations.norm(dim=1)
                    residual_norms = class_residuals.norm(dim=1)
                    component_diagnostics = {
                        "initialized": False,
                        "class_translation_mean_norm": float(
                            translation_norms.mean().item()
                        ),
                        "class_translation_max_norm": float(
                            translation_norms.max().item()
                        ),
                        "class_residual_mean_norm": float(
                            residual_norms.mean().item()
                        ),
                        "class_residual_max_norm": float(
                            residual_norms.max().item()
                        ),
                    }
                trainer.model = current_model
                new_full = (
                    introduction_prototypes
                    if introduction_prototypes is not None
                    else _full_introduction_prototypes(
                        trainer,
                        current_model,
                        session_id,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                    )
                )
                transported = torch.cat(
                    [transported_old.cpu(), new_full.cpu()], dim=0
                )
                if self.settings.population_mass_transport_enabled:
                    population_mass_transport_bank = torch.cat(
                        [population_mass_old.cpu(), new_full.cpu()], dim=0
                    )
                if self.settings.component_ablation_enabled:
                    class_translation_bank = torch.cat(
                        [class_translation_old.cpu(), new_full.cpu()], dim=0
                    )
                    combined_bank = torch.cat(
                        [combined_old.cpu(), new_full.cpu()], dim=0
                    )
                if self.settings.neighbor_affine_enabled:
                    neighbor_affine_bank = torch.cat(
                        [neighbor_affine_old.cpu(), new_full.cpu()], dim=0
                    )

            canonical_direct_bank: Tensor | None = None
            canonical_reference_diagnostics: dict[str, Any] | None = None
            if self.settings.canonical_reference_enabled:
                if canonical_reference_model is None:
                    raise RuntimeError("canonical reference model is missing")
                if session_id == 0:
                    canonical_new = _full_introduction_prototypes(
                        trainer,
                        canonical_reference_model,
                        session_id,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                    )
                    canonical_reference_bank = canonical_new.cpu()
                    canonical_direct_bank = canonical_reference_bank.clone()
                    canonical_reference_diagnostics = {
                        "initialized": True,
                        "reference_session": 0,
                        "support_count": 0,
                        "fit_residual": None,
                        "linear_identity_deviation": 0.0,
                        "bias_norm": 0.0,
                    }
                else:
                    if (
                        previous_checkpoint is None
                        or canonical_reference_bank is None
                    ):
                        raise RuntimeError(
                            "canonical direct transport lacks prior state"
                        )
                    trainer.memory = ExemplarMemory.from_state_dict(
                        previous_checkpoint["memory"]
                    )
                    canonical_support = _paired_support_features(
                        trainer,
                        canonical_reference_model,
                        current_model,
                        session_id - 1,
                        horizontal_flip=(
                            self.settings.support_horizontal_flip
                        ),
                    )
                    canonical_old, canonical_mapping, canonical_residual = (
                        affine_ridge_transport(
                            canonical_reference_bank,
                            canonical_support.old_fit_features,
                            canonical_support.current_fit_features,
                            ridge=self.settings.canonical_reference_ridge,
                        )
                    )
                    canonical_new = _full_introduction_prototypes(
                        trainer,
                        canonical_reference_model,
                        session_id,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                    )
                    canonical_reference_bank = torch.cat(
                        [canonical_reference_bank, canonical_new.cpu()], dim=0
                    )
                    canonical_direct_bank = torch.cat(
                        [canonical_old.cpu(), canonical_new.cpu()], dim=0
                    )
                    canonical_linear = canonical_mapping[:-1]
                    canonical_bias = canonical_mapping[-1]
                    canonical_identity = torch.eye(
                        canonical_linear.shape[0],
                        dtype=canonical_linear.dtype,
                        device=canonical_linear.device,
                    )
                    canonical_reference_diagnostics = {
                        "initialized": False,
                        "reference_session": 0,
                        "support_count": (
                            canonical_support.fit_support_count
                        ),
                        "fit_residual": canonical_residual,
                        "affine_ridge": (
                            self.settings.canonical_reference_ridge
                        ),
                        "linear_identity_deviation": float(
                            (canonical_linear - canonical_identity).norm().item()
                            / canonical_linear.shape[0] ** 0.5
                        ),
                        "bias_norm": float(canonical_bias.norm().item()),
                    }

            if transported is None or transported.shape[0] != seen:
                raise RuntimeError(
                    f"CMPT prototype bank has invalid shape at S{session_id}: "
                    f"{None if transported is None else tuple(transported.shape)}"
                )
            baseline_means = checkpoint["class_means"].detach().cpu()
            # The original exploration that established the CMPT gain changes
            # only old-class prototypes.  Current-session classes keep exactly
            # the same exemplar NME means as the control; their full-data
            # introduction prototypes are stored only for transport after they
            # become old.  Consequently S0 is identical by construction.
            cmpt_means = build_old_class_cmpt_means(
                baseline_means,
                transported,
                old_class_count,
            )
            population_mass_transport_means: Tensor | None = None
            if self.settings.population_mass_transport_enabled:
                if (
                    population_mass_transport_bank is None
                    or population_mass_transport_bank.shape
                    != baseline_means.shape
                ):
                    raise RuntimeError(
                        "population-mass prototype bank has invalid shape"
                    )
                population_mass_transport_means = (
                    build_old_class_cmpt_means(
                        baseline_means,
                        population_mass_transport_bank,
                        old_class_count,
                    )
                )
            moment_calibrated_evaluation_means: dict[str, Tensor] = {}
            if self.settings.moment_calibrated_affine_enabled:
                if len(moment_calibrated_states) != len(CALIBRATION_MODES):
                    raise RuntimeError(
                        "moment-calibrated affine states are incomplete"
                    )
                for mode in CALIBRATION_MODES:
                    state = moment_calibrated_states[mode]
                    if state.prototypes.shape != baseline_means.shape:
                        raise RuntimeError(
                            f"moment-calibrated affine {mode} bank has an "
                            "invalid shape"
                        )
                    moment_calibrated_evaluation_means[mode] = (
                        build_old_class_cmpt_means(
                            baseline_means,
                            state.prototypes,
                            old_class_count,
                        )
                    )
            moment_transport_evaluation_means: Tensor | None = None
            if self.settings.moment_transport_enabled:
                if (
                    moment_direct_bank is None
                    or moment_direct_bank.shape != baseline_means.shape
                ):
                    raise RuntimeError(
                        "moment-transport prototype bank has invalid shape"
                    )
                moment_transport_evaluation_means = (
                    build_old_class_cmpt_means(
                        baseline_means,
                        moment_direct_bank,
                        old_class_count,
                    )
                )
            moment_grid_evaluation_means: dict[str, Tensor] = {}
            if self.settings.moment_grid_enabled:
                if len(moment_grid_direct_banks) != len(moment_grid_specs):
                    raise RuntimeError(
                        "moment-grid prototype banks are incomplete"
                    )
                for key, _, _, _ in moment_grid_specs:
                    direct_bank = moment_grid_direct_banks[key]
                    if direct_bank.shape != baseline_means.shape:
                        raise RuntimeError(
                            f"moment-grid bank {key} has invalid shape"
                        )
                    moment_grid_evaluation_means[key] = (
                        build_old_class_cmpt_means(
                            baseline_means,
                            direct_bank,
                            old_class_count,
                        )
                    )
            canonical_reference_means: Tensor | None = None
            if self.settings.canonical_reference_enabled:
                if (
                    canonical_direct_bank is None
                    or canonical_direct_bank.shape != baseline_means.shape
                ):
                    raise RuntimeError(
                        "canonical direct prototype bank has invalid shape"
                    )
                canonical_reference_means = build_old_class_cmpt_means(
                    baseline_means,
                    canonical_direct_bank,
                    old_class_count,
                )
            persistent_quadrature_means: Tensor | None = None
            persistent_quadrature_diagnostics: dict[str, Any] | None = None
            if self.settings.persistent_quadrature_enabled:
                # Transport fitting temporarily installs the previous memory.
                # Quadrature fitting and application require the complete
                # current memory and the stable identities selected when each
                # class was introduced.
                trainer.memory = ExemplarMemory.from_state_dict(
                    checkpoint["memory"]
                )
                new_weights, introduced_diagnostics = (
                    _fit_new_persistent_quadrature_weights(
                        trainer,
                        current_model,
                        session_id,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                        uniform_ridge=(
                            self.settings.persistent_quadrature_ridge
                        ),
                        max_iterations=(
                            self.settings.persistent_quadrature_iterations
                        ),
                        multiview=(
                            self.settings.persistent_quadrature_multiview
                        ),
                        previous_model=quadrature_previous_model,
                        previous_model_view=(
                            self.settings.persistent_quadrature_previous_model_view
                        ),
                    )
                )
                persistent_quadrature_weights.extend(
                    row.detach().cpu().clone() for row in new_weights
                )
                if len(persistent_quadrature_weights) != seen:
                    raise RuntimeError(
                        "persistent quadrature weight bank has invalid size"
                    )
                weight_bank = torch.stack(persistent_quadrature_weights)
                weighted_all, quadrature_support_count = (
                    _persistent_quadrature_prototypes(
                        trainer,
                        current_model,
                        session_id,
                        weight_bank,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                    )
                )
                persistent_quadrature_means = build_old_class_cmpt_means(
                    baseline_means,
                    weighted_all,
                    old_class_count,
                )
                persistent_quadrature_diagnostics = {
                    "introduced_classes": introduced_diagnostics,
                    "stored_weight_class_count": len(
                        persistent_quadrature_weights
                    ),
                    "old_replaced_class_count": old_class_count,
                    "current_memory_support_count": (
                        quadrature_support_count
                    ),
                }
            herding_extrapolated_means: Tensor | None = None
            herding_extrapolation_diagnostics: dict[str, Any] | None = None
            if self.settings.herding_extrapolation_enabled:
                # CMPT transition fitting temporarily installs the previous
                # memory.  HTE must use the current checkpoint's complete
                # ordered memory, including the just-introduced classes.
                trainer.memory = ExemplarMemory.from_state_dict(
                    checkpoint["memory"]
                )
                recomputed_nme, herding_extrapolated_means, support_count = (
                    _herding_extrapolated_prototypes(
                        trainer,
                        current_model,
                        session_id,
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                        full_prefix=self.settings.herding_full_prefix,
                        reference_prefix=(
                            self.settings.herding_reference_prefix
                        ),
                    )
                )
                nme_parity = float(
                    (recomputed_nme - baseline_means).abs().max().item()
                )
                if (
                    self.settings.strict_parity
                    and nme_parity > self.settings.parity_tolerance
                ):
                    raise RuntimeError(
                        f"{self.settings.learner} S{session_id} ordered-memory "
                        f"prototype parity failed: max_abs={nme_parity:.3e}"
                    )
                herding_extrapolation_diagnostics = {
                    "support_count": support_count,
                    "classes": seen,
                    "full_prefix": self.settings.herding_full_prefix,
                    "reference_prefix": (
                        self.settings.herding_reference_prefix
                    ),
                    "nme_parity_max_abs": nme_parity,
                }
            class_translation_means: Tensor | None = None
            combined_means: Tensor | None = None
            neighbor_affine_means: Tensor | None = None
            if self.settings.component_ablation_enabled:
                if (
                    class_translation_bank is None
                    or combined_bank is None
                ):
                    raise RuntimeError(
                        "component ablation prototype banks are missing"
                    )
                if not (
                    class_translation_bank.shape == transported.shape
                    and combined_bank.shape == transported.shape
                ):
                    raise RuntimeError(
                        "component ablation prototype-bank shapes differ"
                    )
                class_translation_means = build_old_class_cmpt_means(
                    baseline_means,
                    class_translation_bank,
                    old_class_count,
                )
                combined_means = build_old_class_cmpt_means(
                    baseline_means,
                    combined_bank,
                    old_class_count,
                )
            if self.settings.neighbor_affine_enabled:
                if neighbor_affine_bank is None:
                    raise RuntimeError(
                        "neighbor-affine evaluation prototype bank is missing"
                    )
                if neighbor_affine_bank.shape != transported.shape:
                    raise RuntimeError(
                        "neighbor-affine and global prototype-bank shapes differ"
                    )
                neighbor_affine_means = build_old_class_cmpt_means(
                    baseline_means,
                    neighbor_affine_bank,
                    old_class_count,
                )
            adaptive_estimate: AdaptiveAlphaEstimate | None = None
            adaptive_banks: dict[str, Tensor] = {}
            if self.settings.adaptive_alpha_enabled:
                if session_id == 0:
                    adaptive_banks = {
                        "adaptive_session": baseline_means,
                        "adaptive_raw_class": baseline_means,
                        "adaptive_shrunken_class": baseline_means,
                    }
                else:
                    if paired_support is None:
                        raise RuntimeError(
                            "adaptive alpha transition lacks paired support"
                        )
                    if self.settings.transport != "affine_ridge":
                        raise ValueError(
                            "adaptive alpha currently requires affine_ridge"
                        )
                    adaptive_estimate = estimate_adaptive_alphas(
                        paired_support,
                        num_classes=old_class_count,
                        folds=self.settings.adaptive_alpha_folds,
                        ridge=self.settings.affine_ridge,
                    )
                    session_alphas = torch.full(
                        (old_class_count,),
                        adaptive_estimate.session_alpha,
                    )
                    adaptive_banks = {
                        "adaptive_session": build_old_class_adaptive_means(
                            baseline_means,
                            transported,
                            old_class_count,
                            session_alphas,
                        ),
                        "adaptive_raw_class": (
                            build_old_class_adaptive_means(
                                baseline_means,
                                transported,
                                old_class_count,
                                adaptive_estimate.raw_class_alphas,
                            )
                        ),
                        "adaptive_shrunken_class": (
                            build_old_class_adaptive_means(
                                baseline_means,
                                transported,
                                old_class_count,
                                adaptive_estimate.shrunken_class_alphas,
                            )
                        ),
                    }
            full_mean_all_seen: Tensor | None = None
            full_mean_old_only: Tensor | None = None
            full_mean_diagnostics: dict[str, Any] | None = None
            if self.settings.full_mean_oracle_enabled:
                full_mean_all_seen, full_training_image_count = (
                    _full_current_prototypes(
                        trainer,
                        current_model,
                        session_id,
                        trainer.protocol.seen_classes(session_id),
                        horizontal_flip=(
                            self.settings.prototype_horizontal_flip
                        ),
                    )
                )
                if full_mean_all_seen.shape != baseline_means.shape:
                    raise RuntimeError(
                        "full-mean oracle and NME prototype banks differ"
                    )
                full_mean_old_only = baseline_means.detach().clone()
                if old_class_count > 0:
                    full_mean_old_only[:old_class_count] = (
                        full_mean_all_seen[:old_class_count]
                    )
                per_class_distance = 1.0 - F.cosine_similarity(
                    F.normalize(baseline_means.float(), dim=1),
                    F.normalize(full_mean_all_seen.float(), dim=1),
                    dim=1,
                )
                old_distance = (
                    float(per_class_distance[:old_class_count].mean().item())
                    if old_class_count > 0
                    else 0.0
                )
                full_mean_diagnostics = {
                    "full_training_image_count": full_training_image_count,
                    "old_class_count": old_class_count,
                    "old_mean_cosine_distance": old_distance,
                    "all_seen_mean_cosine_distance": float(
                        per_class_distance.mean().item()
                    ),
                    "per_class_cosine_distances": [
                        float(value) for value in per_class_distance.tolist()
                    ],
                }
            geometric_alphas = torch.empty(0)
            geometric_diagnostics: dict[str, Any] | None = None
            geometric_bank: Tensor | None = None
            if self.settings.class_geometric_oracle_enabled:
                if session_id == 0:
                    geometric_bank = baseline_means
                    geometric_diagnostics = {
                        "initialized": True,
                        "full_old_training_image_count": 0,
                        "baseline_mean_cosine_distance": None,
                        "cmpt_mean_cosine_distance": None,
                        "oracle_mean_cosine_distance": None,
                    }
                else:
                    full_current_old, full_image_count = (
                        _full_current_old_prototypes(
                            trainer,
                            current_model,
                            session_id,
                            horizontal_flip=(
                                self.settings.prototype_horizontal_flip
                            ),
                        )
                    )
                    geometric_alphas, distances = (
                        class_geometric_oracle_alphas(
                            baseline_means[:old_class_count],
                            cmpt_means[:old_class_count],
                            full_current_old,
                            self.settings.oracle_alpha_grid,
                        )
                    )
                    geometric_bank = build_old_class_adaptive_means(
                        baseline_means,
                        transported,
                        old_class_count,
                        geometric_alphas,
                    )
                    geometric_diagnostics = {
                        "initialized": False,
                        "full_old_training_image_count": full_image_count,
                        **distances,
                    }
            test_dataset = trainer.data.cumulative_test_dataset(session_id)
            test_loader = trainer._loader(
                test_dataset,
                shuffle=False,
                session_id=session_id + 11000,
            )
            prototype_interpolation: dict[str, dict[str, Any]] | None = None
            adaptive_metrics: dict[str, dict[str, Any]] | None = None
            component_metrics: dict[str, dict[str, Any]] | None = None
            neighbor_affine_metrics: dict[str, Any] | None = None
            herding_extrapolation_metrics: dict[str, Any] | None = None
            persistent_quadrature_metrics: dict[str, Any] | None = None
            canonical_reference_metrics: dict[str, Any] | None = None
            population_mass_transport_metrics: dict[str, Any] | None = None
            moment_transport_metrics: dict[str, Any] | None = None
            moment_grid_metrics: dict[str, dict[str, Any]] | None = None
            moment_calibrated_metrics: dict[str, dict[str, Any]] | None = None
            if (
                self.settings.prototype_interpolation_alphas
                or self.settings.adaptive_alpha_enabled
                or self.settings.class_geometric_oracle_enabled
                or self.settings.full_mean_oracle_enabled
                or self.settings.component_ablation_enabled
                or self.settings.neighbor_affine_enabled
                or self.settings.herding_extrapolation_enabled
                or self.settings.persistent_quadrature_enabled
                or self.settings.canonical_reference_enabled
                or self.settings.population_mass_transport_enabled
                or self.settings.moment_calibrated_affine_enabled
                or self.settings.moment_transport_enabled
            ):
                evaluation_banks: dict[str, Tensor] = {
                    "__baseline": baseline_means,
                    "__cmpt": cmpt_means,
                }
                if self.settings.component_ablation_enabled:
                    if (
                        class_translation_means is None
                        or combined_means is None
                    ):
                        raise RuntimeError(
                            "component ablation evaluation banks are missing"
                        )
                    evaluation_banks.update(
                        {
                            "class_translation": class_translation_means,
                            "combined_cmpt": combined_means,
                        }
                    )
                if self.settings.neighbor_affine_enabled:
                    if neighbor_affine_means is None:
                        raise RuntimeError(
                            "neighbor-affine evaluation bank is missing"
                        )
                    evaluation_banks["neighbor_affine"] = (
                        neighbor_affine_means
                    )
                if self.settings.herding_extrapolation_enabled:
                    if herding_extrapolated_means is None:
                        raise RuntimeError(
                            "herding extrapolation prototype bank is missing"
                        )
                    evaluation_banks["herding_extrapolation"] = (
                        herding_extrapolated_means
                    )
                if self.settings.persistent_quadrature_enabled:
                    if persistent_quadrature_means is None:
                        raise RuntimeError(
                            "persistent quadrature prototype bank is missing"
                        )
                    evaluation_banks["persistent_quadrature"] = (
                        persistent_quadrature_means
                    )
                if self.settings.canonical_reference_enabled:
                    if canonical_reference_means is None:
                        raise RuntimeError(
                            "canonical reference evaluation bank is missing"
                        )
                    evaluation_banks["canonical_reference"] = (
                        canonical_reference_means
                    )
                if self.settings.population_mass_transport_enabled:
                    if population_mass_transport_means is None:
                        raise RuntimeError(
                            "population-mass evaluation bank is missing"
                        )
                    evaluation_banks["population_mass_transport"] = (
                        population_mass_transport_means
                    )
                if self.settings.moment_calibrated_affine_enabled:
                    evaluation_banks.update(
                        {
                            f"moment_calibrated:{mode}": means
                            for mode, means in (
                                moment_calibrated_evaluation_means.items()
                            )
                        }
                    )
                if self.settings.moment_transport_enabled:
                    if moment_transport_evaluation_means is None:
                        raise RuntimeError(
                            "moment-transport evaluation bank is missing"
                        )
                    evaluation_banks["moment_transport"] = (
                        moment_transport_evaluation_means
                    )
                if self.settings.moment_grid_enabled:
                    evaluation_banks.update(
                        {
                            f"moment_grid:{key}": means
                            for key, means in (
                                moment_grid_evaluation_means.items()
                            )
                        }
                    )
                evaluation_banks.update(adaptive_banks)
                if geometric_bank is not None:
                    evaluation_banks["class_geometric_oracle"] = (
                        geometric_bank
                    )
                if self.settings.full_mean_oracle_enabled:
                    if (
                        full_mean_old_only is None
                        or full_mean_all_seen is None
                    ):
                        raise RuntimeError(
                            "full-mean oracle prototype banks are missing"
                        )
                    evaluation_banks["full_mean_old_only"] = (
                        full_mean_old_only
                    )
                    evaluation_banks["full_mean_all_seen"] = (
                        full_mean_all_seen
                    )
                evaluation_banks.update(
                    {
                        _alpha_key(alpha): (
                            build_old_class_interpolated_means(
                                baseline_means,
                                transported,
                                old_class_count,
                                alpha,
                            )
                        )
                        for alpha in (
                            self.settings.prototype_interpolation_alphas
                        )
                    }
                )
                evaluated_banks = _evaluate_nme_banks(
                    current_model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    evaluation_banks,
                    center_strength=self.settings.center_strength,
                    horizontal_flip_query=(
                        self.settings.query_horizontal_flip
                    ),
                )
                baseline = copy.deepcopy(evaluated_banks["__baseline"])
                cmpt = copy.deepcopy(evaluated_banks["__cmpt"])
                if self.settings.prototype_interpolation_alphas:
                    prototype_interpolation = {
                        _alpha_key(alpha): copy.deepcopy(
                            evaluated_banks[_alpha_key(alpha)]
                        )
                        for alpha in (
                            self.settings.prototype_interpolation_alphas
                        )
                    }
                if self.settings.adaptive_alpha_enabled:
                    adaptive_metrics = {
                        "session": copy.deepcopy(
                            evaluated_banks["adaptive_session"]
                        ),
                        "raw_class": copy.deepcopy(
                            evaluated_banks["adaptive_raw_class"]
                        ),
                        "shrunken_class": copy.deepcopy(
                            evaluated_banks["adaptive_shrunken_class"]
                        ),
                    }
                if self.settings.component_ablation_enabled:
                    component_metrics = {
                        "class_translation": copy.deepcopy(
                            evaluated_banks["class_translation"]
                        ),
                        "combined_cmpt": copy.deepcopy(
                            evaluated_banks["combined_cmpt"]
                        ),
                    }
                if self.settings.neighbor_affine_enabled:
                    neighbor_affine_metrics = copy.deepcopy(
                        evaluated_banks["neighbor_affine"]
                    )
                if self.settings.herding_extrapolation_enabled:
                    herding_extrapolation_metrics = copy.deepcopy(
                        evaluated_banks["herding_extrapolation"]
                    )
                if self.settings.persistent_quadrature_enabled:
                    persistent_quadrature_metrics = copy.deepcopy(
                        evaluated_banks["persistent_quadrature"]
                    )
                if self.settings.canonical_reference_enabled:
                    canonical_reference_metrics = copy.deepcopy(
                        evaluated_banks["canonical_reference"]
                    )
                if self.settings.population_mass_transport_enabled:
                    population_mass_transport_metrics = copy.deepcopy(
                        evaluated_banks["population_mass_transport"]
                    )
                if self.settings.moment_calibrated_affine_enabled:
                    moment_calibrated_metrics = {
                        mode: copy.deepcopy(
                            evaluated_banks[f"moment_calibrated:{mode}"]
                        )
                        for mode in CALIBRATION_MODES
                    }
                if self.settings.moment_transport_enabled:
                    moment_transport_metrics = copy.deepcopy(
                        evaluated_banks["moment_transport"]
                    )
                if self.settings.moment_grid_enabled:
                    moment_grid_metrics = {
                        key: copy.deepcopy(
                            evaluated_banks[f"moment_grid:{key}"]
                        )
                        for key, _, _, _ in moment_grid_specs
                    }
                geometric_metrics = (
                    copy.deepcopy(
                        evaluated_banks["class_geometric_oracle"]
                    )
                    if self.settings.class_geometric_oracle_enabled
                    else None
                )
                full_mean_metrics = (
                    {
                        "old_only": copy.deepcopy(
                            evaluated_banks["full_mean_old_only"]
                        ),
                        "all_seen": copy.deepcopy(
                            evaluated_banks["full_mean_all_seen"]
                        ),
                    }
                    if self.settings.full_mean_oracle_enabled
                    else None
                )
            else:
                geometric_metrics = None
                full_mean_metrics = None
                herding_extrapolation_metrics = None
                persistent_quadrature_metrics = None
                canonical_reference_metrics = None
                population_mass_transport_metrics = None
                moment_calibrated_metrics = None
                moment_transport_metrics = None
                moment_grid_metrics = None
                baseline = evaluate_nme(
                    current_model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    baseline_means,
                    center_strength=self.settings.center_strength,
                    horizontal_flip_query=(
                        self.settings.query_horizontal_flip
                    ),
                ).to_dict()
                cmpt = evaluate_nme(
                    current_model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    cmpt_means,
                    center_strength=self.settings.center_strength,
                    horizontal_flip_query=(
                        self.settings.query_horizontal_flip
                    ),
                ).to_dict()
            native = self._evaluate_native(
                current_model,
                test_loader,
                trainer.device,
                old_class_count,
                baseline,
            )
            stored = _checkpoint_session_metric(checkpoint, session_id)
            parity_error = float(baseline["accuracy"]) - float(
                stored["accuracy"]
            )
            if (
                self.settings.strict_parity
                and abs(parity_error) > self.settings.parity_tolerance
            ):
                raise RuntimeError(
                    f"{self.settings.learner} S{session_id} NME parity failed: "
                    f"recomputed={baseline['accuracy']:.8f}, "
                    f"stored={float(stored['accuracy']):.8f}, "
                    f"error={parity_error:+.3e}"
                )

            adaptive_alpha_record: dict[str, Any] | None = None
            if self.settings.adaptive_alpha_enabled:
                if adaptive_metrics is None:
                    raise RuntimeError("adaptive alpha metrics are missing")
                if adaptive_estimate is None:
                    empty = torch.empty(0)
                    session_stats = _alpha_statistics(empty)
                    raw_stats = _alpha_statistics(empty)
                    shrunken_stats = _alpha_statistics(empty)
                    diagnostics: dict[str, Any] = {
                        "initialized": True,
                        "folds": None,
                        "old_exemplar_count": 0,
                        "memory_uncertainties": [],
                        "transport_uncertainties": [],
                        "jackknife_variances": [],
                        "shrinkage_credibilities": [],
                        "between_class_variance": None,
                    }
                else:
                    session_stats = _alpha_statistics(
                        torch.full(
                            (old_class_count,),
                            adaptive_estimate.session_alpha,
                        )
                    )
                    raw_stats = _alpha_statistics(
                        adaptive_estimate.raw_class_alphas
                    )
                    shrunken_stats = _alpha_statistics(
                        adaptive_estimate.shrunken_class_alphas
                    )
                    diagnostics = {
                        "initialized": False,
                        "folds": adaptive_estimate.folds,
                        "old_exemplar_count": (
                            paired_support.exemplar_count
                            if paired_support is not None
                            else 0
                        ),
                        "memory_uncertainties": [
                            float(value)
                            for value in adaptive_estimate.memory_uncertainties.tolist()
                        ],
                        "transport_uncertainties": [
                            float(value)
                            for value in adaptive_estimate.transport_uncertainties.tolist()
                        ],
                        "jackknife_variances": [
                            float(value)
                            for value in adaptive_estimate.jackknife_variances.tolist()
                        ],
                        "shrinkage_credibilities": [
                            float(value)
                            for value in adaptive_estimate.shrinkage_credibilities.tolist()
                        ],
                        "between_class_variance": (
                            adaptive_estimate.between_class_variance
                        ),
                    }
                adaptive_alpha_record = {
                    "session": {
                        **session_stats,
                        "metrics": adaptive_metrics["session"],
                    },
                    "raw_class": {
                        **raw_stats,
                        "metrics": adaptive_metrics["raw_class"],
                    },
                    "shrunken_class": {
                        **shrunken_stats,
                        "metrics": adaptive_metrics["shrunken_class"],
                    },
                    "diagnostics": diagnostics,
                }

            class_geometric_record: dict[str, Any] | None = None
            if self.settings.class_geometric_oracle_enabled:
                if geometric_metrics is None or geometric_diagnostics is None:
                    raise RuntimeError("class-geometric oracle result is missing")
                class_geometric_record = {
                    **_alpha_statistics(geometric_alphas),
                    "metrics": geometric_metrics,
                    "diagnostics": geometric_diagnostics,
                    "oracle_only": True,
                    "uses_full_old_training_data": session_id > 0,
                }

            full_mean_record: dict[str, Any] | None = None
            if self.settings.full_mean_oracle_enabled:
                if full_mean_metrics is None or full_mean_diagnostics is None:
                    raise RuntimeError("full-mean oracle result is missing")
                full_mean_record = {
                    **full_mean_metrics,
                    "diagnostics": full_mean_diagnostics,
                    "oracle_only": True,
                    "uses_full_training_data": True,
                    "uses_test_labels_for_selection": False,
                }

            record = {
                "checkpoint": str(checkpoint_path),
                "session_id": session_id,
                "seen_classes": seen,
                "native": native,
                "baseline": baseline,
                "cmpt": cmpt,
                "delta_accuracy": float(cmpt["accuracy"])
                - float(baseline["accuracy"]),
                "stored_nme_accuracy": float(stored["accuracy"]),
                "parity_error": parity_error,
                "transport_diagnostics": transport_diagnostics,
                "session_elapsed_seconds": time.perf_counter()
                - session_started,
            }
            if self.settings.component_ablation_enabled:
                if component_metrics is None:
                    raise RuntimeError(
                        "component ablation metrics are missing"
                    )
                record["class_translation"] = component_metrics[
                    "class_translation"
                ]
                record["combined_cmpt"] = component_metrics[
                    "combined_cmpt"
                ]
                record["component_ablation"] = {
                    "class_translation_delta_vs_nme": float(
                        component_metrics["class_translation"]["accuracy"]
                    )
                    - float(baseline["accuracy"]),
                    "combined_delta_vs_nme": float(
                        component_metrics["combined_cmpt"]["accuracy"]
                    )
                    - float(baseline["accuracy"]),
                    "combined_delta_vs_global": float(
                        component_metrics["combined_cmpt"]["accuracy"]
                    )
                    - float(cmpt["accuracy"]),
                    "combined_delta_vs_class": float(
                        component_metrics["combined_cmpt"]["accuracy"]
                    )
                    - float(
                        component_metrics["class_translation"]["accuracy"]
                    ),
                    "diagnostics": component_diagnostics,
                }
            if self.settings.neighbor_affine_enabled:
                if neighbor_affine_metrics is None:
                    raise RuntimeError("neighbor-affine metrics are missing")
                record["neighbor_affine"] = neighbor_affine_metrics
                record["neighbor_affine_comparison"] = {
                    "delta_vs_nme": float(
                        neighbor_affine_metrics["accuracy"]
                    )
                    - float(baseline["accuracy"]),
                    "delta_vs_global": float(
                        neighbor_affine_metrics["accuracy"]
                    )
                    - float(cmpt["accuracy"]),
                    "diagnostics": neighbor_affine_diagnostics,
                }
            if prototype_interpolation is not None:
                record["prototype_interpolation"] = (
                    prototype_interpolation
                )
            if adaptive_alpha_record is not None:
                record["adaptive_alpha"] = adaptive_alpha_record
            if class_geometric_record is not None:
                record["class_geometric_oracle"] = class_geometric_record
            if full_mean_record is not None:
                record["full_mean_oracle"] = full_mean_record
            if self.settings.herding_extrapolation_enabled:
                if (
                    herding_extrapolation_metrics is None
                    or herding_extrapolation_diagnostics is None
                ):
                    raise RuntimeError(
                        "herding extrapolation result is missing"
                    )
                record["herding_extrapolation"] = (
                    herding_extrapolation_metrics
                )
                record["herding_extrapolation_diagnostics"] = (
                    herding_extrapolation_diagnostics
                )
            if self.settings.persistent_quadrature_enabled:
                if (
                    persistent_quadrature_metrics is None
                    or persistent_quadrature_diagnostics is None
                ):
                    raise RuntimeError(
                        "persistent quadrature result is missing"
                    )
                record["persistent_quadrature"] = (
                    persistent_quadrature_metrics
                )
                record["persistent_quadrature_diagnostics"] = (
                    persistent_quadrature_diagnostics
                )
            if self.settings.canonical_reference_enabled:
                if (
                    canonical_reference_metrics is None
                    or canonical_reference_diagnostics is None
                ):
                    raise RuntimeError(
                        "canonical reference result is missing"
                    )
                record["canonical_reference"] = (
                    canonical_reference_metrics
                )
                record["canonical_reference_diagnostics"] = (
                    canonical_reference_diagnostics
                )
            if self.settings.population_mass_transport_enabled:
                if population_mass_transport_metrics is None:
                    raise RuntimeError(
                        "population-mass transport result is missing"
                    )
                record["population_mass_transport"] = (
                    population_mass_transport_metrics
                )
                record["population_mass_transport_diagnostics"] = (
                    population_mass_transport_diagnostics
                )
            if self.settings.moment_calibrated_affine_enabled:
                if moment_calibrated_metrics is None:
                    raise RuntimeError(
                        "moment-calibrated affine metrics are missing"
                    )
                if len(moment_calibrated_diagnostics) != len(
                    CALIBRATION_MODES
                ):
                    raise RuntimeError(
                        "moment-calibrated affine diagnostics are incomplete"
                    )
                record["moment_calibrated_affine"] = (
                    moment_calibrated_metrics
                )
                record["moment_calibrated_affine_diagnostics"] = (
                    moment_calibrated_diagnostics
                )
            if self.settings.moment_transport_enabled:
                if moment_transport_metrics is None:
                    raise RuntimeError(
                        "moment-transport result is missing"
                    )
                record["moment_transport"] = moment_transport_metrics
                record["moment_transport_diagnostics"] = (
                    moment_transport_diagnostics
                )
            if self.settings.moment_grid_enabled:
                if moment_grid_metrics is None:
                    raise RuntimeError("moment-grid metrics are missing")
                if len(moment_grid_validation) != len(moment_grid_specs):
                    raise RuntimeError(
                        "moment-grid validation metrics are incomplete"
                    )
                record["moment_transport_grid"] = moment_grid_metrics
                record["moment_transport_grid_validation"] = (
                    moment_grid_validation
                )
            records.append(record)
            elapsed = time.perf_counter() - started
            dump_json(
                self._partial_payload(
                    records, status="running", elapsed_seconds=elapsed
                ),
                output,
            )
            adaptive_progress = ""
            if adaptive_alpha_record is not None:
                adaptive_progress = (
                    ", Adapt-S="
                    f"{100.0 * float(adaptive_alpha_record['session']['metrics']['accuracy']):.3f}"
                    ", Adapt-C="
                    f"{100.0 * float(adaptive_alpha_record['raw_class']['metrics']['accuracy']):.3f}"
                    ", Adapt-CS="
                    f"{100.0 * float(adaptive_alpha_record['shrunken_class']['metrics']['accuracy']):.3f}"
                )
            oracle_progress = ""
            if class_geometric_record is not None:
                oracle_progress = (
                    ", Geo-Oracle="
                    f"{100.0 * float(class_geometric_record['metrics']['accuracy']):.3f}"
                )
            full_mean_progress = ""
            if full_mean_record is not None:
                full_mean_progress = (
                    ", Full-Old="
                    f"{100.0 * float(full_mean_record['old_only']['accuracy']):.3f}"
                    ", Full-All="
                    f"{100.0 * float(full_mean_record['all_seen']['accuracy']):.3f}"
                )
            component_progress = ""
            if self.settings.component_ablation_enabled:
                assert component_metrics is not None
                component_progress = (
                    ", Class-T="
                    f"{100.0 * float(component_metrics['class_translation']['accuracy']):.3f}"
                    ", Combined="
                    f"{100.0 * float(component_metrics['combined_cmpt']['accuracy']):.3f}"
                )
            neighbor_progress = ""
            if self.settings.neighbor_affine_enabled:
                assert neighbor_affine_metrics is not None
                neighbor_progress = (
                    ", Local-Affine="
                    f"{100.0 * float(neighbor_affine_metrics['accuracy']):.3f}"
                )
            herding_progress = ""
            if self.settings.herding_extrapolation_enabled:
                assert herding_extrapolation_metrics is not None
                herding_progress = (
                    ", HTE="
                    f"{100.0 * float(herding_extrapolation_metrics['accuracy']):.3f}"
                )
            quadrature_progress = ""
            if self.settings.persistent_quadrature_enabled:
                assert persistent_quadrature_metrics is not None
                quadrature_progress = (
                    ", PEQ="
                    f"{100.0 * float(persistent_quadrature_metrics['accuracy']):.3f}"
                )
            canonical_progress = ""
            if self.settings.canonical_reference_enabled:
                assert canonical_reference_metrics is not None
                canonical_progress = (
                    ", CRPT="
                    f"{100.0 * float(canonical_reference_metrics['accuracy']):.3f}"
                )
            population_mass_progress = ""
            if self.settings.population_mass_transport_enabled:
                assert population_mass_transport_metrics is not None
                population_mass_progress = (
                    ", PM-Affine="
                    f"{100.0 * float(population_mass_transport_metrics['accuracy']):.3f}"
                )
            moment_progress = ""
            if self.settings.moment_transport_enabled:
                assert moment_transport_metrics is not None
                moment_progress = (
                    ", Moment-T="
                    f"{100.0 * float(moment_transport_metrics['accuracy']):.3f}"
                )
            moment_calibrated_progress = ""
            if self.settings.moment_calibrated_affine_enabled:
                assert moment_calibrated_metrics is not None
                moment_calibrated_progress = (
                    ", MC-Mean="
                    f"{100.0 * float(moment_calibrated_metrics['mean']['accuracy']):.3f}"
                    ", MC-Second="
                    f"{100.0 * float(moment_calibrated_metrics['second']['accuracy']):.3f}"
                    ", MC-Both="
                    f"{100.0 * float(moment_calibrated_metrics['combined']['accuracy']):.3f}"
                )
            self.progress(
                f"{self.settings.learner} S{session_id}: "
                f"Native={100.0 * float(native['accuracy']):.3f}, "
                f"NME={100.0 * float(baseline['accuracy']):.3f}, "
                f"CMPT={100.0 * float(cmpt['accuracy']):.3f}, "
                f"delta={100.0 * record['delta_accuracy']:+.3f} pp"
                f"{adaptive_progress}"
                f"{oracle_progress}"
                f"{full_mean_progress}"
                f"{component_progress}"
                f"{neighbor_progress}"
                f"{herding_progress}"
                f"{quadrature_progress}"
                f"{canonical_progress}"
                f"{population_mass_progress}"
                f"{moment_calibrated_progress}"
                f"{moment_progress}"
            )
            previous_checkpoint = checkpoint
            if quadrature_previous_model is not None:
                del quadrature_previous_model
            del current_model

        payload = self._partial_payload(
            records,
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
        )
        dump_json(payload, output)
        return payload
