from __future__ import annotations

from pathlib import Path

from sacil.data import CIFAR100DataModule, ClassOrderProtocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiment_configs"
    / "class_orders"
    / "cifar100_b50_t10_afc_order1.json"
)


def test_cifar_protocol_is_complete_permutation() -> None:
    protocol = ClassOrderProtocol.from_json(PROTOCOL)
    assert protocol.num_sessions == 11
    assert len(protocol.class_order) == 100
    assert set(protocol.class_order) == set(range(100))
    assert protocol.session(0).size == 50
    assert all(protocol.session(index).size == 5 for index in range(1, 11))
    assert protocol.seen_classes(10) == protocol.class_order


def test_real_cifar_session_counts_and_no_future_class_leakage() -> None:
    protocol = ClassOrderProtocol.from_json(PROTOCOL)
    data = CIFAR100DataModule(ROOT / "datasets", protocol)
    base_train = data.new_train_dataset(0)
    first_increment = data.new_train_dataset(1)
    base_test = data.cumulative_test_dataset(0)
    first_test = data.cumulative_test_dataset(1)
    assert len(base_train) == 25_000
    assert len(first_increment) == 2_500
    assert len(base_test) == 5_000
    assert len(first_test) == 5_500
    assert {
        base_train[index]["original_target"]
        for index in range(len(base_train))
    } == set(protocol.classes_for_session(0))
    assert {
        first_increment[index]["original_target"]
        for index in range(len(first_increment))
    } == set(protocol.classes_for_session(1))


def test_original_and_incremental_label_mapping_roundtrip() -> None:
    protocol = ClassOrderProtocol.from_json(PROTOCOL)
    for incremental, original in enumerate(protocol.class_order):
        assert protocol.incremental_label(original) == incremental
        assert protocol.original_label(incremental) == original

