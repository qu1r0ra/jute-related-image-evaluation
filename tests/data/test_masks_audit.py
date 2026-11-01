import csv
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.cli import main
from jute_disease.data.masks import SCENE_TAGS, SceneThresholds
from jute_disease.data.masks_audit import (
    CandidateImage,
    FixtureMaskBackend,
    evaluate_masks_audit,
    run_masks_audit,
)


def _thresholds() -> SceneThresholds:
    ids = tuple(f"dev-{index}" for index in range(100))
    sources = tuple(
        (image_id, hashlib.sha256(f"source-{index}".encode()).hexdigest())
        for index, image_id in enumerate(ids)
    )
    return SceneThresholds(
        background_fraction_min=0.5,
        color_uniformity_min=0.9,
        development_ids=ids,
        development_sources=sources,
        development_digest="a" * 64,
        score=1.0,
        mask_model=FixtureMaskBackend.model,
        mask_version=FixtureMaskBackend.version,
    )


def _image(path: Path, scene: str, index: int) -> None:
    pixels = np.full((8, 8, 3), 245, dtype=np.uint8)
    if scene == "field":
        for y in range(8):
            pixels[y, :, :] = 30 + y * 25, 60 + y * 17, 10 + y * 9
    if scene == "unknown":
        pixels[1:8, 1:8] = (0, 180, 0)
    else:
        pixels[2:6, 2:6] = (0, 180, 0)
    pixels[0, 0] = (245, 245, index % 256)
    pixels[0, 1] = (245, 245, index // 256)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def _candidates(tmp_path: Path) -> list[CandidateImage]:
    result = []
    counts = {"clean-background": 10, "field": 6, "unknown": 4}
    index = 0
    for tag, count in counts.items():
        for _ in range(count):
            path = tmp_path / f"candidate-{index:02d}.png"
            _image(path, tag, index)
            result.append(CandidateImage(f"candidate-{index:02d}", path))
            index += 1
    return result


def _fill_ratings(manifest: dict, out: Path) -> None:
    with (out / "agent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "overlay_path", "mask_quality")
        )
        writer.writeheader()
        for index, row in enumerate(manifest["records"]):
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "overlay_path": row["overlay_path"],
                    "mask_quality": ("pass", "minor", "fail")[index % 3],
                }
            )
    with (out / "owner.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audit_id", "overlay_path", "mask_quality", "scene_tag"),
        )
        writer.writeheader()
        for index, audit_id in enumerate(manifest["owner_audit_ids"]):
            row = next(
                row for row in manifest["records"] if row["audit_id"] == audit_id
            )
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "overlay_path": row["overlay_path"],
                    "mask_quality": ("pass", "minor", "fail")[index % 3],
                    "scene_tag": row["predicted_scene"],
                }
            )


def test_stratified_audit_oversamples_rare_scenes_and_blinds_sheets(
    tmp_path: Path,
) -> None:
    manifest = run_masks_audit(
        _candidates(tmp_path),
        _thresholds(),
        FixtureMaskBackend(),
        tmp_path / "audit",
        sample_count=12,
        owner_count=6,
        seed=10,
    )

    assert manifest["sampling"]["population_by_scene"] == {
        "clean-background": 10,
        "field": 6,
        "unknown": 4,
    }
    assert manifest["sampling"]["sampled_by_scene"] == {
        "clean-background": 4,
        "field": 4,
        "unknown": 4,
    }
    id_scene_order = [row["predicted_scene"] for row in manifest["records"]]
    assert (
        sum(
            id_scene_order[index] != id_scene_order[index - 1]
            for index in range(1, len(id_scene_order))
        )
        > len(SCENE_TAGS) - 1
    )
    owner_columns = next(csv.reader((tmp_path / "audit" / "owner.csv").open()))
    agent_columns = next(csv.reader((tmp_path / "audit" / "agent.csv").open()))
    assert owner_columns == ["audit_id", "overlay_path", "mask_quality", "scene_tag"]
    assert agent_columns == ["audit_id", "overlay_path", "mask_quality"]
    metadata = json.loads(
        (tmp_path / "audit" / manifest["records"][0]["metadata_path"]).read_text()
    )
    assert metadata["model"] == FixtureMaskBackend.model
    assert metadata["model_version"] == FixtureMaskBackend.version
    assert metadata["rubric"]["version"] == "leaf-mask-v1"

    _fill_ratings(manifest, tmp_path / "audit")
    report = evaluate_masks_audit(
        manifest,
        tmp_path / "audit" / "agent.csv",
        tmp_path / "audit" / "owner.csv",
        resamples=200,
        agent_rater={"name": "fixture-agent", "version": "test"},
    )
    confusion = report["scene_tag"]["confusion_matrix"]
    assert confusion["labels"] == list(SCENE_TAGS)
    assert len(confusion["counts"]) == 3
    assert all(len(row) == 3 for row in confusion["counts"])
    assert sum(map(sum, confusion["counts"])) == 6
    assert report["mask_quality"]["fail_rate"]["population"] == 20


@pytest.mark.parametrize("rating", ["", "invalid"])
def test_evaluation_rejects_invalid_unpaired_agent_rating(
    tmp_path: Path, rating: str
) -> None:
    manifest = run_masks_audit(
        _candidates(tmp_path),
        _thresholds(),
        FixtureMaskBackend(),
        tmp_path / "audit",
        sample_count=12,
        owner_count=6,
        seed=10,
    )
    _fill_ratings(manifest, tmp_path / "audit")
    unpaired_id = next(
        row["audit_id"]
        for row in manifest["records"]
        if row["audit_id"] not in manifest["owner_audit_ids"]
    )
    agent_path = tmp_path / "audit" / "agent.csv"
    with agent_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["audit_id"] == unpaired_id:
            row["mask_quality"] = rating
    with agent_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "overlay_path", "mask_quality")
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        ValueError, match=f"invalid mask_quality rating for {unpaired_id}"
    ):
        evaluate_masks_audit(
            manifest,
            agent_path,
            tmp_path / "audit" / "owner.csv",
            resamples=200,
            agent_rater={"name": "fixture-agent", "version": "test"},
        )


def test_audit_sample_includes_each_nonempty_stratum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="every nonempty scene stratum"):
        run_masks_audit(
            _candidates(tmp_path),
            _thresholds(),
            FixtureMaskBackend(),
            tmp_path / "audit",
            sample_count=2,
            owner_count=1,
        )


def test_audit_rejects_development_image_content_overlap(tmp_path: Path) -> None:
    path = tmp_path / "same.png"
    _image(path, "clean-background", 1)
    image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    thresholds = _thresholds()
    sources = list(thresholds.development_sources)
    sources[0] = (sources[0][0], image_hash)
    thresholds = SceneThresholds(
        background_fraction_min=thresholds.background_fraction_min,
        color_uniformity_min=thresholds.color_uniformity_min,
        development_ids=thresholds.development_ids,
        development_sources=tuple(sources),
        development_digest=thresholds.development_digest,
        score=thresholds.score,
        mask_model=thresholds.mask_model,
        mask_version=thresholds.mask_version,
    )
    candidates = [CandidateImage(f"candidate-{index}", path) for index in range(3)]

    with pytest.raises(ValueError, match="development image content"):
        run_masks_audit(
            candidates,
            thresholds,
            FixtureMaskBackend(),
            tmp_path / "audit",
            sample_count=3,
            owner_count=1,
        )


def test_sam2_loader_is_local_only_and_fixture_never_loads_sam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    import types

    calls = []
    processor_inputs = []

    class FakeTensor:
        def __init__(self, value: np.ndarray | None = None) -> None:
            self.value = value

        def cpu(self) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            assert self.value is not None
            return self.value

    class FakeScalar:
        def item(self) -> float:
            return 0.91

    class FakeScores:
        def reshape(self, *_: object) -> FakeScores:
            return self

        def argmax(self) -> FakeScalar:
            return FakeScalar()

        def __getitem__(self, index: int) -> FakeScalar:
            assert index == 0
            return FakeScalar()

    class FakeMasks:
        def __getitem__(self, index: tuple[int, int]) -> FakeTensor:
            assert index == (0, 0)
            return FakeTensor(np.ones((8, 8), dtype=np.uint8))

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeModel:
            calls.append(("model", path, kwargs))
            return cls()

        def to(self, device: str) -> FakeModel:
            assert device == "cpu"
            return self

        def eval(self) -> FakeModel:
            return self

        def __call__(self, **_: object) -> object:
            return types.SimpleNamespace(
                pred_masks=FakeTensor(), iou_scores=FakeScores()
            )

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeProcessor:
            calls.append(("processor", path, kwargs))
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, FakeTensor]:
            processor_inputs.append(kwargs)
            return {"original_sizes": FakeTensor()}

        def post_process_masks(
            self, masks: FakeTensor, original_sizes: FakeTensor
        ) -> list[FakeMasks]:
            assert isinstance(masks, FakeTensor)
            assert isinstance(original_sizes, FakeTensor)
            return [FakeMasks()]

    checkpoint = tmp_path / "sam"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
    transformer_stub = types.ModuleType("transformers")
    transformer_stub.__version__ = "fixture-version"
    transformer_stub.Sam2Model = FakeModel
    transformer_stub.Sam2Processor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformer_stub)
    torch_stub = types.ModuleType("torch")
    torch_stub.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    from jute_disease.data.masks import Sam2Refiner

    backend = Sam2Refiner(checkpoint)
    assert backend.version.startswith("fixture-version+")
    assert (
        backend.checkpoint_sha256
        == hashlib.sha256(
            b"model.safetensors"
            + hashlib.sha256(b"fixture weights").hexdigest().encode()
        ).hexdigest()
    )
    assert len(calls) == 2
    assert all(call[1] == str(checkpoint.resolve()) for call in calls)
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)

    pixels = np.full((8, 8, 3), (200, 150, 100), dtype=np.uint8)
    pixels[2:6, 2:6] = (0, 180, 0)
    prediction = backend.predict(Image.fromarray(pixels))
    assert processor_inputs[0]["input_points"] == [[[[3.5, 3.5]]]]
    assert processor_inputs[0]["input_labels"] == [[[1]]]
    assert prediction.diagnostics["threshold_sam_iou"] == pytest.approx(0.25)
    assert prediction.diagnostics["checkpoint_sha256"] == backend.checkpoint_sha256

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("weight-free fixture must not construct SAM2")

    monkeypatch.setattr("jute_disease.cli.Sam2Refiner", forbidden)
    report = main(["data", "masks-fixture", "--output", str(tmp_path / "fixture")])
    assert report is None
    fixture_report = json.loads(capsys.readouterr().out)
    assert fixture_report["fixture_mode"] is True
    assert (
        fixture_report["evidence_status"]
        == "synthetic fixture only; not an audit result"
    )
    assert fixture_report["rating_counts"]["owner_scene"]
