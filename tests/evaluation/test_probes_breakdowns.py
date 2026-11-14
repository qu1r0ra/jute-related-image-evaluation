"""Probe and breakdown rules that the CLI fixture is too small to reach."""

import numpy as np

from jute_disease.evaluation.breakdowns import (
    class_scene_cells,
    scene_summaries,
    stratified_sample,
)
from jute_disease.evaluation.probes import (
    border_probe,
    family_permutation,
    frame_leaf_share,
    frame_mask,
    permutation_null,
    permutation_test,
    probe_result,
)
from jute_disease.evaluation.scores import PredictionSet

CLASSES = ("a", "b")


def _system(n_per_class: int, accuracy_every: int = 1) -> PredictionSet:
    ids, truth, groups = [], [], []
    for label in CLASSES:
        for i in range(n_per_class):
            ids.append(f"{label}{i:04d}")
            truth.append(label)
            groups.append(f"{label}-fam{i // 5}")
    predicted = tuple(
        t if i % accuracy_every == 0 else CLASSES[1 - CLASSES.index(t)]
        for i, t in enumerate(truth)
    )
    return PredictionSet(
        CLASSES,
        tuple(ids),
        tuple(truth),
        tuple(groups),
        tuple("src" for _ in ids),
        {0: predicted},
    )


def test_frame_mask_and_leaf_share() -> None:
    valid = np.zeros((20, 20), dtype=bool)
    valid[2:18, 2:18] = True
    frame = frame_mask(valid, 3)
    assert frame.sum() == 16 * 16 - 10 * 10
    assert not frame[:2].any()
    leaf = np.zeros_like(valid)
    leaf[2:5, 2:18] = True
    share = frame_leaf_share(leaf, valid, 3)
    assert share == 3 * 16 / frame.sum()


def test_border_headline_withheld_when_a_class_has_few_retained_images() -> None:
    system = _system(40)
    share = dict.fromkeys(system.image_ids, 0.0)
    report = border_probe(
        system, share, {}, mask_audit_passed=True, resamples=50, permutations=20
    )
    assert report.headline is None
    assert "fewer than 50" in (report.headline_withheld or "")
    assert report.unrestricted.n == 80
    assert report.limit


def test_border_headline_withheld_when_mask_audit_failed() -> None:
    system = _system(60)
    share = dict.fromkeys(system.image_ids, 0.0)
    report = border_probe(
        system, share, {}, mask_audit_passed=False, resamples=50, permutations=20
    )
    assert report.headline is None
    assert "audit failed" in (report.headline_withheld or "")


def test_border_headline_reported_on_low_leaf_share_images() -> None:
    system = _system(60)
    share = {i: (0.5 if i.endswith("9") else 0.0) for i in system.image_ids}
    report = border_probe(
        system, share, {}, mask_audit_passed=True, resamples=50, permutations=20
    )
    assert report.headline is not None
    assert report.headline.n == report.retained_images < 120
    assert min(report.retained_class_counts.values()) >= 50


def test_permutation_null_is_near_chance_and_rule_detects_signal() -> None:
    system = _system(60, accuracy_every=2)
    null = permutation_null(system, 200)
    assert 0.3 < float(np.percentile(null, 95)) < 0.7
    perfect = _system(60)
    assert permutation_null(perfect, 50).mean() < 0.75


MIXED_FAMILIES = (
    ("a", "b"),
    ("a", "b"),
    ("a", "a", "b"),
    ("a", "b", "b"),
    ("a", "a", "b"),
    ("a", "b", "b"),
    ("a",),
    ("b",),
    ("b",),
    ("a",),
)


def _mixed_system(
    rng: np.random.Generator, repeats: int, predicted: str = "random"
) -> PredictionSet:
    ids, truth, groups = [], [], []
    for r in range(repeats):
        for f, labels in enumerate(MIXED_FAMILIES):
            for j, label in enumerate(labels):
                ids.append(f"r{r}f{f}i{j}")
                truth.append(label)
                groups.append(f"r{r}f{f}")
    if predicted == "truth":
        guess = tuple(truth)
    else:
        guess = tuple(str(x) for x in rng.choice(list(CLASSES), size=len(ids)))
    return PredictionSet(
        CLASSES,
        tuple(ids),
        tuple(truth),
        tuple(groups),
        tuple("src" for _ in ids),
        {0: guess},
    )


def test_family_permutation_moves_whole_label_multisets_between_families() -> None:
    system = _mixed_system(np.random.default_rng(0), 3)
    families = sorted(set(system.groups))
    original = {
        f: tuple(
            sorted(
                t for t, g in zip(system.truth, system.groups, strict=True) if g == f
            )
        )
        for f in families
    }
    rng = np.random.default_rng(5)
    moved = 0
    for _ in range(30):
        labels = family_permutation(system, rng)
        permuted = {
            f: tuple(
                sorted(
                    system.classes[k]
                    for k, g in zip(labels, system.groups, strict=True)
                    if g == f
                )
            )
            for f in families
        }
        assert sorted(permuted.values()) == sorted(original.values())
        for f in families:
            assert len(permuted[f]) == len(original[f])
        moved += permuted != original
    assert moved > 0


def test_probe_null_has_nominal_size_on_mixed_label_families() -> None:
    rng = np.random.default_rng(11)
    simulations = 300
    rejected = 0
    for k in range(simulations):
        system = _mixed_system(rng, 2)
        test = permutation_test(system, 99, seed=k)
        rejected += test.p_value < 0.05
    assert 0.02 <= rejected / simulations <= 0.09


def test_probe_null_detects_label_information_on_mixed_label_families() -> None:
    system = _mixed_system(np.random.default_rng(3), 4, predicted="truth")
    test = permutation_test(system, 199, seed=1)
    assert test.p_value <= 0.01


def test_probe_decision_uses_the_permutation_p_value() -> None:
    signal = probe_result(
        _mixed_system(np.random.default_rng(3), 4, predicted="truth"),
        "found",
        "none",
        resamples=50,
        permutations=199,
    )
    assert signal.detected
    assert signal.p_value < 0.05
    assert signal.statement == "found"
    noise = probe_result(
        _mixed_system(np.random.default_rng(4), 4),
        "found",
        "none",
        resamples=50,
        permutations=199,
    )
    assert noise.detected == (noise.p_value < 0.05)
    assert noise.macro_f1.lower <= noise.macro_f1.estimate <= noise.macro_f1.upper


def test_scene_macro_f1_needs_thirty_per_class() -> None:
    system = _system(40)
    scenes = {i: ("big" if int(i[1:]) < 35 else "small") for i in system.image_ids}
    summaries = {s.scene: s for s in scene_summaries(system, scenes)}
    assert summaries["big"].macro_f1 == 1.0
    assert summaries["small"].macro_f1 is None
    assert "below 30" in (summaries["small"].reason or "")


def test_class_scene_cells_report_counts_and_wilson_intervals() -> None:
    system = _system(10, accuracy_every=2)
    scenes = dict.fromkeys(system.image_ids, "s")
    cells = {c.label: c for c in class_scene_cells(system, scenes)}
    assert cells["a"].n == 10
    assert cells["a"].accuracy.lower < cells["a"].accuracy.upper


def test_stratified_sample_is_proportional_and_deterministic() -> None:
    ids = [f"i{n:04d}" for n in range(1000)]
    labels = ["x" if n < 800 else "y" for n in range(1000)]
    sample = stratified_sample(ids, labels, 100, 7)
    assert len(sample) == 100
    assert sum(i < "i0800" for i in sample) == 80
    assert sample == stratified_sample(ids, labels, 100, 7)
    assert sample != stratified_sample(ids, labels, 100, 8)
