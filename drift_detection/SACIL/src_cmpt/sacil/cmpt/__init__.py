"""Checkpoint-frozen Co-Moving Prototype Transport experiments."""

from .evaluator import (
    CMPTCheckpointEvaluator,
    CMPTExperimentSettings,
    NativeClassifierSpec,
    TrajectoryAudit,
    adaptive_alpha_from_uncertainties,
    audit_checkpoint_trajectory,
    build_old_class_adaptive_means,
    build_old_class_cmpt_means,
    build_old_class_interpolated_means,
    class_geometric_oracle_alphas,
    discover_checkpoint_paths,
    empirical_bayes_shrink_alphas,
    resolve_native_classifier,
)
from .herding_extrapolation import herding_prefix_extrapolated_means
from .moment_transport import (
    MomentTransportMap,
    PolynomialKernelMomentTransportMap,
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
from .persistent_quadrature import (
    fit_multiview_persistent_quadrature_weights,
    fit_persistent_quadrature_weights,
    project_probability_simplex,
    weighted_class_prototypes,
)

__all__ = [
    "CMPTCheckpointEvaluator",
    "CMPTExperimentSettings",
    "NativeClassifierSpec",
    "TrajectoryAudit",
    "adaptive_alpha_from_uncertainties",
    "audit_checkpoint_trajectory",
    "build_old_class_adaptive_means",
    "build_old_class_cmpt_means",
    "build_old_class_interpolated_means",
    "class_geometric_oracle_alphas",
    "discover_checkpoint_paths",
    "empirical_bayes_shrink_alphas",
    "resolve_native_classifier",
    "herding_prefix_extrapolated_means",
    "MomentTransportMap",
    "PolynomialKernelMomentTransportMap",
    "apply_low_rank_moment_transport",
    "apply_polynomial_kernel_moment_transport",
    "class_first_second_moments",
    "fit_low_rank_moment_transport",
    "fit_low_rank_moment_transport_grid",
    "fit_polynomial_kernel_moment_transport",
    "CALIBRATION_MODES",
    "expand_exemplar_weights",
    "fit_moment_calibration_weights",
    "transport_first_second_moments_affine",
    "fit_persistent_quadrature_weights",
    "fit_multiview_persistent_quadrature_weights",
    "project_probability_simplex",
    "weighted_class_prototypes",
]
