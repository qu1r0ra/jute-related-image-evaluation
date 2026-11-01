"""The declared benchmark matrix, recipe, plan order and run identities."""

import json
from dataclasses import replace

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import (
    Benchmark,
    FitKind,
    build_plan,
    fit_identity,
    load_benchmark,
)


@pytest.fixture(scope="module")
def benchmark() -> Benchmark:
    return load_benchmark(BENCHMARK_CONFIG)


def test_recipe_matches_the_frozen_specification(benchmark) -> None:
    recipe = benchmark.recipe
    assert recipe.learning_rate == 3e-4
    assert recipe.weight_decay == 0.01
    assert recipe.max_epochs == 50
    assert recipe.early_stopping_patience == 8
    assert recipe.scheduler_patience == 3
    assert recipe.scheduler_factor == 0.1
    assert recipe.image_size == 256
    assert recipe.seeds == (42, 1337, 7)
    assert recipe.source_seed == 42
    assert recipe.compile_model is False
    assert recipe.num_workers == 0
    assert recipe.weighted_sampler is True
    assert recipe.effective_batch == 32


def test_ten_classical_fits_cover_every_estimator_on_both_feature_sets(
    benchmark,
) -> None:
    classical = [f for f in benchmark.fits() if f.kind is FitKind.CLASSICAL]
    assert len(classical) == 10
    assert {(f.estimator, f.features) for f in classical} == {
        (e, x) for e in ("gnb", "knn", "lr", "rf", "svm") for x in ("raw", "crafted")
    }
    assert {f.seed for f in classical} == {42}


def test_twelve_target_fits_across_four_systems_and_three_seeds(benchmark) -> None:
    targets = [f for f in benchmark.fits() if f.kind is FitKind.TARGET]
    assert len(targets) == 12
    assert {f.system for f in targets} == {
        "mobilenet_v2_direct",
        "efficientnet_b5_direct",
        "mobilenet_v2_plantvillage",
        "mobilenet_v2_plantvillage_plantdoc",
    }
    for system in {f.system for f in targets}:
        assert {f.seed for f in targets if f.system == system} == {42, 1337, 7}


def test_initialization_recipes_and_batching(benchmark) -> None:
    by_id = {f.fit_id: f for f in benchmark.fits()}
    mobile = by_id["cnn/mobilenet_v2_direct/seed-42"]
    b5 = by_id["cnn/efficientnet_b5_direct/seed-42"]
    assert mobile.model == "mobilenetv2_100.ra_in1k"
    assert (mobile.micro_batch, mobile.accumulation) == (32, 1)
    assert b5.model == "efficientnet_b5.sw_in12k_ft_in1k"
    assert (b5.micro_batch, b5.accumulation) == (8, 4)


def test_one_shared_source_chain_feeds_both_domain_arms(benchmark) -> None:
    by_id = {f.fit_id: f for f in benchmark.fits()}
    pv, pd = by_id["source/plantvillage"], by_id["source/plantdoc"]
    assert (pv.kind, pv.stage, pv.seed, pv.upstream) == (
        FitKind.SOURCE,
        "plantvillage",
        42,
        None,
    )
    assert (pd.stage, pd.seed, pd.upstream) == ("plantdoc", 42, "source/plantvillage")
    for seed in (42, 1337, 7):
        assert (
            by_id[f"cnn/mobilenet_v2_plantvillage/seed-{seed}"].upstream
            == "source/plantvillage"
        )
        assert (
            by_id[f"cnn/mobilenet_v2_plantvillage_plantdoc/seed-{seed}"].upstream
            == "source/plantdoc"
        )
        assert by_id[f"cnn/mobilenet_v2_direct/seed-{seed}"].upstream is None


def test_plan_order_is_core_classical_plantvillage_then_plantdoc(benchmark) -> None:
    plan = build_plan(benchmark)
    assert [s.name for s in plan] == ["core", "classical", "plantvillage", "plantdoc"]
    core, classical, pv, pd = plan
    assert [f.system for f in core.fits] == (
        ["mobilenet_v2_direct"] * 3 + ["efficientnet_b5_direct"] * 3
    )
    assert classical.fits[0].kind is FitKind.PREPARE
    assert len(classical.fits) == 11
    assert [f.fit_id for f in pv.fits][0] == "source/plantvillage"
    assert [f.fit_id for f in pd.fits][0] == "source/plantdoc"
    assert len(pv.fits) == len(pd.fits) == 4


def test_every_declared_fit_is_in_exactly_one_stage(benchmark) -> None:
    plan_ids = [f.fit_id for stage in build_plan(benchmark) for f in stage.fits]
    assert len(plan_ids) == len(set(plan_ids))
    assert set(plan_ids) == {f.fit_id for f in benchmark.fits()}


def test_identity_is_stable_and_sensitive_to_every_input(benchmark) -> None:
    fit = benchmark.fits()[0]
    base = fit_identity(
        fit, benchmark.recipe, manifests={"target": "m1"}, upstream_sha256=None
    )
    assert base == fit_identity(
        fit, benchmark.recipe, manifests={"target": "m1"}, upstream_sha256=None
    )
    assert base != fit_identity(
        fit, benchmark.recipe, manifests={"target": "m2"}, upstream_sha256=None
    )
    assert base != fit_identity(
        fit, benchmark.recipe, manifests={"target": "m1"}, upstream_sha256="abc"
    )
    assert base != fit_identity(
        replace(fit, seed=fit.seed + 1),
        benchmark.recipe,
        manifests={"target": "m1"},
        upstream_sha256=None,
    )
    assert base != fit_identity(
        fit,
        replace(benchmark.recipe, learning_rate=1e-3),
        manifests={"target": "m1"},
        upstream_sha256=None,
    )
    assert base != fit_identity(
        fit,
        benchmark.recipe,
        manifests={"target": "m1"},
        upstream_sha256=None,
        environment={"pretrained": True},
    )


def test_unknown_configuration_keys_are_rejected(tmp_path) -> None:
    text = BENCHMARK_CONFIG.read_text(encoding="utf-8").replace(
        '"weight_decay": 0.01', '"weight_decay": 0.01, "mystery": 1', 1
    )
    path = tmp_path / "benchmark.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="mystery"):
        load_benchmark(path)


def test_compilation_cannot_be_enabled(tmp_path) -> None:
    text = BENCHMARK_CONFIG.read_text(encoding="utf-8").replace(
        '"compile_model": false', '"compile_model": true', 1
    )
    path = tmp_path / "benchmark.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="compile"):
        load_benchmark(path)


def test_each_kind_of_step_has_its_own_timeout(benchmark) -> None:
    limits = benchmark.limits
    assert limits.timeout(FitKind.PREPARE) == limits.prepare_seconds
    assert limits.timeout(FitKind.CLASSICAL) == limits.classical_seconds
    assert limits.timeout(FitKind.SOURCE) == limits.source_seconds
    assert limits.timeout(FitKind.TARGET) == limits.target_seconds


@pytest.mark.parametrize("value", ["0", "-5", "true", '"1800"', "1e999"])
def test_a_timeout_must_be_a_finite_positive_number(tmp_path, value: str) -> None:
    raw = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    text = json.dumps(raw).replace(
        f'"target_seconds": {raw["queue"]["target_seconds"]}',
        f'"target_seconds": {value}',
        1,
    )
    path = tmp_path / "benchmark.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="target_seconds"):
        load_benchmark(path)


def test_the_queue_declares_no_deadline_or_budget() -> None:
    raw = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    assert set(raw["queue"]) == {
        "prepare_seconds",
        "classical_seconds",
        "source_seconds",
        "target_seconds",
    }
