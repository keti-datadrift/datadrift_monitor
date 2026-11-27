from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedMethodContract:
    """Static provenance and comparison contract for one Table-1 adapter.

    ``reference_path`` is documentation metadata.  The unified runner never
    imports it, adds it to ``sys.path``, or starts it as a subprocess.
    """

    name: str
    implementation_module: str
    reference_path: str
    reference_origin: str
    training_classifier: str
    evaluation_classifier: str = "nme"
    backbone_family: str = "resnet32"
    reference_only: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONTRACTS = {
    "joint": UnifiedMethodContract(
        "joint",
        "sacil.engine.table1_trainer",
        "",
        "in-repo upper-bound control",
        "linear_fc",
    ),
    "finetune": UnifiedMethodContract(
        "finetune",
        "sacil.methods.finetune",
        "ref_codes/00_frameworks/PyCIL/models/finetune.py",
        "official PyCIL",
        "linear_fc_new_slice_ce",
    ),
    "replay": UnifiedMethodContract(
        "replay",
        "sacil.methods.replay_ce",
        "ref_codes/00_frameworks/PyCIL/models/replay.py",
        "official PyCIL",
        "linear_fc_replay_ce",
    ),
    "icarl": UnifiedMethodContract(
        "icarl",
        "sacil.methods.icarl",
        "ref_codes/00_frameworks/PyCIL/models/icarl.py",
        "official PyCIL",
        "linear_fc_ce_plus_old_softmax_kd",
    ),
    "podnet": UnifiedMethodContract(
        "podnet",
        "sacil.methods.podnet",
        "ref_codes/00_frameworks/PyCIL/models/podnet.py",
        "official PyCIL",
        "multi_proxy_nca",
    ),
    "afc": UnifiedMethodContract(
        "afc",
        "sacil.methods.afc",
        "ref_codes/02_stability_plasticity_selective_preservation/AFC/inclearn/models/afc.py",
        "author-official GitHub",
        "multi_proxy_nca",
    ),
    "create": UnifiedMethodContract(
        "create",
        "sacil.methods.create",
        "ref_codes/01_geometry_topology_preservation/CREATE/models/create.py",
        "author-official GitHub",
        "classwise_autoencoder",
        evaluation_classifier="native_reconstruction_error",
    ),
    "fgp": UnifiedMethodContract(
        "fgp",
        "sacil.methods.fgp",
        "ref_codes/01_geometry_topology_preservation/FGP-ICL/models/fgp.py",
        "author-official GitHub",
        "rectified_cosine_bce",
    ),
    "cscct": UnifiedMethodContract(
        "cscct",
        "sacil.methods.cscct",
        "ref_codes/01_geometry_topology_preservation/CSCCT/trainer/incremental_icarl.py",
        "author-official GitHub",
        "split_cosine_fc",
    ),
    "casper": UnifiedMethodContract(
        "casper",
        "sacil.methods.casper",
        "ref_codes/01_geometry_topology_preservation/CaSpeR-IL/models/icarl.py",
        "author-official GitHub",
        "icarl_bce_plus_spectral",
    ),
    "sacil": UnifiedMethodContract(
        "sacil",
        "sacil.methods.sacil_v0",
        "mds/ideas/Locally_Relaxed_Hierarchical_Anchor_CIL_idea.md",
        "SACIL proposal",
        "prototype_ce",
    ),
}


SUPPORTED_UNIFIED_METHODS = frozenset(_CONTRACTS)


def unified_method_contract(name: str) -> UnifiedMethodContract:
    key = str(name).lower()
    try:
        return _CONTRACTS[key]
    except KeyError as error:
        raise ValueError(f"unsupported unified method: {name}") from error


def validate_annotation1_config(config: dict[str, Any]) -> None:
    """Reject protocol drift in the controlled CIFAR-100 Table-1 suite."""

    data = config.get("data", {})
    model = config.get("model", {})
    memory = config.get("memory", {})
    evaluation = config.get("evaluation", {})
    if str(data.get("name", "")).lower() != "cifar100":
        raise ValueError("Annotation 1 requires CIFAR-100")
    if not str(data.get("protocol", "")).replace("\\", "/").endswith(
        "cifar100_b50_t10_afc_order1.json"
    ):
        raise ValueError("Annotation 1 requires the fixed AFC/iCaRL class order")
    if data.get("color_jitter") is not True:
        raise ValueError("Annotation 1 requires the shared augmentation pipeline")
    if "resnet18" in str(model.get("backbone", "")).lower():
        raise ValueError("Annotation 1 requires a CIFAR ResNet-32 backbone")
    if "32" not in str(model.get("backbone", "")).lower():
        raise ValueError("model.backbone must identify its ResNet-32 adaptation")
    if str(memory.get("mode", "")) != "per_class":
        raise ValueError("Annotation 1 requires fixed exemplars per class")
    if int(memory.get("exemplars_per_class", -1)) != 20:
        raise ValueError("Annotation 1 requires 20 exemplars per class")
    if str(memory.get("selection", "")) != "icarl_herding":
        raise ValueError("Annotation 1 requires shared iCaRL herding")
    method = str(config.get("method", {}).get("name", "")).lower()
    expected_evaluator = "native" if method == "create" else "nme"
    if str(evaluation.get("classifier", "")).lower() != expected_evaluator:
        raise ValueError(
            f"{method} requires evaluator={expected_evaluator} in the "
            "controlled Table-1 contract"
        )


def validate_annotation1_protocol(protocol: Any) -> None:
    sessions = [protocol.session(index) for index in range(protocol.num_sessions)]
    sizes = [session.size for session in sessions]
    if sizes != [50] + [5] * 10:
        raise ValueError("Annotation 1 requires CIFAR-100 B50 + 10x5")
