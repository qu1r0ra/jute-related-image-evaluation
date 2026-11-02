"""The declared benchmark matrix, recipe, plan order and run identities."""

import json
from dataclasses import replace

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import (
    Benchmark,
    DataInputs,
    FitKind,
    Workspace,
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
    assert recipe.seeds == (42, 1337, 7, 21, 84)
    assert recipe.source_seeds == (42, 1337, 7)
    assert recipe.compile_model is False
    assert recipe.precision == "bf16-mixed"
    assert recipe.num_workers == 8
    assert recipe.weighted_sampler is True
    assert recipe.effective_batch == 32


def test_cpu_smoke_workspace_uses_zero_workers(tmp_path) -> None:
    arguments = {
        "benchmark_path": BENCHMARK_CONFIG,
        "data": DataInputs(
            image_root=tmp_path,
            target_dir=tmp_path,
            sources={},
            policy_dir=tmp_path / "policies",
        ),
        "runs_dir": tmp_path / "runs",
        "cache_dir": tmp_path / "cache",
        "pretrained": False,
        "accelerator": "auto",
    }

    official = Workspace(**arguments, require_cuda=True)
    smoke = Workspace(**arguments, require_cuda=False)

    assert official.benchmark.recipe.num_workers == 8
    assert smoke.benchmark.recipe.num_workers == 0


def test_ten_classical_fits_cover_every_estimator_on_both_feature_sets(
    benchmark,
) -> None:
    classical = [f for f in benchmark.fits() if f.kind is FitKind.CLASSICAL]
    final_classical = [f for f in classical if f.phase == "final"]
    tuning = [f for f in classical if f.phase == "tune"]
    assert len(final_classical) == 3240
    assert len(tuning) == 30
    assert {(f.estimator, f.features) for f in classical} == {
        (e, x) for e in ("gnb", "knn", "lr", "rf", "svm") for x in ("raw", "crafted")
    }
    assert {f.seed for f in tuning} == {42}
    assert len({f.fit_id for f in classical}) == len(classical)


def test_twelve_target_fits_across_four_systems_and_three_seeds(benchmark) -> None:
    targets = [f for f in benchmark.fits() if f.kind is FitKind.TARGET]
    final_targets = [fit for fit in targets if fit.phase == "final"]
    tuning = [fit for fit in targets if fit.phase == "tune"]
    assert len(final_targets) == 2790
    assert len(tuning) == 12
    assert {(fit.system, fit.pool) for fit in tuning} == {
        (system, pool)
        for system in benchmark.systems
        for pool in benchmark.pools
    }
    assert {f.system for f in targets} == {
        "mobilenet_v2_direct",
        "efficientnet_b5_imagenet",
        "mobilenet_v2_plantvillage",
        "mobilenet_v2_plantvillage_plantdoc",
    }
    assert {
        f.seed
        for f in final_targets
        if f.system == "mobilenet_v2_direct" and f.split == 0
    } == {42, 1337, 7, 21, 84}
    b5 = [f for f in final_targets if f.system == "efficientnet_b5_imagenet"]
    assert len(b5) == 90
    assert {fit.split for fit in b5} == {0}
    assert {fit.view for fit in b5} == {1, 2}
    assert {fit.pool for fit in b5} == {"A", "B", "C"}
    assert {fit.policy for fit in b5} == {
        "image-random",
        "exact-copy",
        "near-duplicate",
    }


def test_initialization_recipes_and_batching(benchmark) -> None:
    by_id = {f.fit_id: f for f in benchmark.fits()}
    mobile = by_id[
        "cnn/mobilenet_v2_direct/pool-A/view-1/image-random/split-0/seed-42"
    ]
    b5 = by_id[
        "cnn/efficientnet_b5_imagenet/pool-A/view-1/image-random/split-0/seed-42"
    ]
    assert mobile.model == "mobilenetv2_100.ra_in1k"
    assert (mobile.micro_batch, mobile.accumulation) == (32, 1)
    assert b5.model == "efficientnet_b5.sw_in12k_ft_in1k"
    assert (b5.micro_batch, b5.accumulation) == (8, 4)


def test_one_shared_source_chain_feeds_both_domain_arms(benchmark) -> None:
    by_id = {f.fit_id: f for f in benchmark.fits()}
    source_tune = by_id["source/tune/plantvillage"]
    pv = by_id["source/plantvillage/seed-42"]
    pd = by_id["source/plantdoc/seed-42"]
    assert (pv.kind, pv.stage, pv.seed, pv.upstream) == (
        FitKind.SOURCE,
        "plantvillage",
        42,
        None,
    )
    assert (source_tune.kind, source_tune.stage, source_tune.phase) == (
        FitKind.SOURCE,
        "plantvillage",
        "tune",
    )
    assert (pd.stage, pd.seed, pd.upstream) == (
        "plantdoc",
        42,
        "source/plantvillage/seed-42",
    )
    target = by_id[
        "cnn/mobilenet_v2_plantvillage_plantdoc/pool-A/view-1/image-random"
        "/split-0/seed-42"
    ]
    assert target.upstream == "source/plantdoc/seed-42"
    assert target.split == 0 and target.seed_index == 0


def test_plan_order_is_core_classical_plantvillage_then_plantdoc(benchmark) -> None:
    plan = build_plan(benchmark)
    assert [s.name for s in plan] == ["core", "classical", "plantvillage", "plantdoc"]
    core, classical, pv, pd = plan
    core_final = [fit for fit in core.fits if fit.phase == "final"]
    assert [f.system for f in core_final] == (
        ["mobilenet_v2_direct"] * 900 + ["efficientnet_b5_imagenet"] * 90
    )
    assert classical.fits[0].kind is FitKind.PREPARE
    assert len(classical.fits) == 3271
    assert pv.fits[0].fit_id == "source/tune/plantvillage"
    assert [f.fit_id for f in pv.fits[1:4]] == [
        "source/plantvillage/seed-42",
        "source/plantvillage/seed-1337",
        "source/plantvillage/seed-7",
    ]
    assert [f.fit_id for f in pd.fits[:3]] == [
        "source/plantdoc/seed-42",
        "source/plantdoc/seed-1337",
        "source/plantdoc/seed-7",
    ]
    assert len(pv.fits) == 907
    assert len(pd.fits) == 906


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
        replace(benchmark.recipe, num_workers=0),
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


def test_identity_binds_configuration_and_code_commit(benchmark) -> None:
    fit = benchmark.fits()[0]
    arguments = {
        "fit": fit,
        "recipe": benchmark.recipe,
        "manifests": {"target": "manifest-sha"},
        "upstream_sha256": None,
        "config_sha256": "config-sha",
        "code_commit": "commit-a",
        "environment_lock_sha256": "lock-sha",
        "preprocessing_sha256": "preprocess-sha",
        "assignment_table_sha256": "assignment-sha",
        "source_checkpoint_ancestry": [
            {
                "fit_id": "source/a",
                "identity": "source-identity",
                "checkpoint_sha256": "checkpoint-sha",
            }
        ],
    }

    base = fit_identity(**arguments)

    assert base == fit_identity(**arguments)
    assert base != fit_identity(**(arguments | {"config_sha256": "config-b"}))
    assert base != fit_identity(**(arguments | {"code_commit": "commit-b"}))
    assert base != fit_identity(**(arguments | {"environment_lock_sha256": "lock-b"}))
    assert base != fit_identity(
        **(arguments | {"preprocessing_sha256": "preprocess-b"})
    )
    assert base != fit_identity(
        **(arguments | {"assignment_table_sha256": "assignment-b"})
    )
    assert base != fit_identity(**(arguments | {"source_checkpoint_ancestry": []}))


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


def test_cuda_recipe_requires_bf16_mixed_precision(tmp_path) -> None:
    text = BENCHMARK_CONFIG.read_text(encoding="utf-8").replace(
        '"precision": "bf16-mixed"', '"precision": "16-mixed"', 1
    )
    path = tmp_path / "benchmark.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="bf16-mixed"):
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
