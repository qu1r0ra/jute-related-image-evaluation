import csv
import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from jute_disease import cli

SHEET_COLUMNS = [
    "item_id",
    "kind",
    "image_a",
    "image_b",
    "overlay_path",
    "rating",
    "mask_quality",
    "scene_tag",
]
HIDDEN = (
    "agent_rating",
    "algorithm_merge",
    "stratum",
    "predicted_scene",
    "inclusion_probability",
    "expansion_weight",
)


class Inputs(NamedTuple):
    cohort_dir: Path
    mask_audit: Path
    cohort: dict[str, Any]


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> Any:
    cli.main(["data", *argv])
    return json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    base = tmp_path_factory.mktemp("kit-inputs")
    cli.main(["data", "masks-fixture", "--output", str(base / "masks")])
    cli.main(
        [
            "data",
            "audit-fixture",
            "--output",
            str(base / "grouping"),
            "--grouping-version",
            "dhash-v1",
        ]
    )
    cohort_dir = base / "grouping"
    cohort = json.loads((cohort_dir / "agent.json").read_text("utf-8"))
    return Inputs(cohort_dir, base / "masks" / "audit", cohort)


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def rate(row: dict[str, str], cohort: dict[str, Any]) -> None:
    if row["kind"] == "mask":
        row["mask_quality"], row["scene_tag"] = "pass", "field"
    else:
        pair = cohort["pairs"][int(row["item_id"].removeprefix("pair:"))]
        row["rating"] = "near-duplicate" if pair["algorithm_merge"] else "different"


def build(capsys: pytest.CaptureFixture[str], inputs: Inputs, out: Path) -> Any:
    return run(
        capsys, "audit-kit", str(inputs.cohort_dir), str(inputs.mask_audit), str(out)
    )


def test_kit_combines_pairs_with_mask_overlays_and_omits_agent_ratings(
    capsys: pytest.CaptureFixture[str], inputs: Inputs, tmp_path: Path
) -> None:
    assert any(p["agent_rating"] for p in inputs.cohort["pairs"])

    summary = build(capsys, inputs, tmp_path / "kit")

    kit = json.loads((tmp_path / "kit" / "kit.json").read_text("utf-8"))
    pair_count = len(inputs.cohort["pairs"])
    assert (summary["mask_items"], summary["pair_items"]) == (120, pair_count)
    kinds = [item["kind"] for item in kit["items"]]
    assert kinds == ["mask"] * 120 + ["pair"] * pair_count
    assert kit["grouping_version"] == "dhash-v1" and kit["hamming_threshold"] == 4
    assert kit["agent"] == inputs.cohort["agent"]
    mask_audit = json.loads((inputs.mask_audit / "audit.json").read_text("utf-8"))
    mask_ids = {item["item_id"].removeprefix("mask:") for item in kit["items"][:120]}
    assert mask_ids == set(mask_audit["owner_audit_ids"])
    for item in kit["items"][:120]:
        assert (tmp_path / "kit" / item["overlay_path"]).is_file()
    pair_ids = {item["item_id"] for item in kit["items"][120:]}
    assert pair_ids == {f"pair:{p['pair_id']}" for p in inputs.cohort["pairs"]}

    run(capsys, "audit-sitting", str(tmp_path / "kit"), "--limit", "5")
    text = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "kit").iterdir()
        if path.suffix in {".json", ".csv"}
    )
    assert not [token for token in HIDDEN if token in text]


def test_kit_is_refused_for_a_cohort_without_grouping_version_or_agent(
    capsys: pytest.CaptureFixture[str], inputs: Inputs, tmp_path: Path
) -> None:
    for missing in ("grouping_version", "agent"):
        cohort_dir = tmp_path / f"no-{missing}"
        cohort_dir.mkdir()
        broken = {k: v for k, v in inputs.cohort.items() if k != missing}
        (cohort_dir / "agent.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(SystemExit, match="audit blocked"):
            cli.main(
                [
                    "data",
                    "audit-kit",
                    str(cohort_dir),
                    str(inputs.mask_audit),
                    str(tmp_path / f"kit-{missing}"),
                ]
            )
        assert not (tmp_path / f"kit-{missing}").exists()


def test_sittings_resume_without_reshowing_rated_items(
    capsys: pytest.CaptureFixture[str], inputs: Inputs, tmp_path: Path
) -> None:
    kit_dir = tmp_path / "kit"
    build(capsys, inputs, kit_dir)

    first = run(capsys, "audit-sitting", str(kit_dir), "--limit", "10")
    sheet = Path(first["sheet"])
    assert sheet.name == "sitting-001.csv" and first["remaining"] > 0
    rows = read_sheet(sheet)
    assert list(rows[0]) == SHEET_COLUMNS and len(rows) == 10
    assert all(not row["rating"] and not row["mask_quality"] for row in rows)
    for row in rows[:6]:
        rate(row, inputs.cohort)
    rows[6]["mask_quality"] = "pass"  # a half-filled mask item stays unrated
    write_sheet(sheet, rows)
    done = {row["item_id"] for row in rows[:6]}

    second = run(capsys, "audit-sitting", str(kit_dir), "--limit", "10")
    again = read_sheet(Path(second["sheet"]))
    assert Path(second["sheet"]).name == "sitting-002.csv"
    assert second["rated"] == 6
    assert not done & {row["item_id"] for row in again}
    assert [row["item_id"] for row in again[:4]] == [r["item_id"] for r in rows[6:]]

    with pytest.raises(SystemExit, match="unrated"):
        cli.main(["data", "audit-collect", str(kit_dir)])

    conflict = read_sheet(Path(second["sheet"]))
    conflict[0] = {**rows[0], "rating": "different", "mask_quality": "fail"}
    write_sheet(Path(second["sheet"]), conflict)
    with pytest.raises(SystemExit, match="differently"):
        cli.main(["data", "audit-sitting", str(kit_dir)])


def test_finished_sittings_collect_into_owner_sheets_that_evaluate(
    capsys: pytest.CaptureFixture[str], inputs: Inputs, tmp_path: Path
) -> None:
    kit_dir = tmp_path / "kit"
    build(capsys, inputs, kit_dir)

    while True:
        step = run(capsys, "audit-sitting", str(kit_dir), "--limit", "100")
        if step["sheet"] is None:
            break
        rows = read_sheet(Path(step["sheet"]))
        for row in rows:
            rate(row, inputs.cohort)
        write_sheet(Path(step["sheet"]), rows)
    collected = run(capsys, "audit-collect", str(kit_dir))

    pairs = read_sheet(kit_dir / "owner.csv")
    masks = read_sheet(kit_dir / "owner-masks.csv")
    assert (collected["pair_ratings"], collected["mask_ratings"]) == (
        len(inputs.cohort["pairs"]),
        120,
    )
    assert list(pairs[0]) == ["pair_id", "image_a", "image_b", "rating"]
    assert list(masks[0]) == ["audit_id", "overlay_path", "mask_quality", "scene_tag"]
    mask_audit = json.loads((inputs.mask_audit / "audit.json").read_text("utf-8"))
    sources = {r["audit_id"]: r["overlay_path"] for r in mask_audit["records"]}
    assert all(row["overlay_path"] == sources[row["audit_id"]] for row in masks)
    report = run(
        capsys,
        "audit-evaluate",
        str(inputs.cohort_dir / "agent.json"),
        str(kit_dir / "owner.csv"),
    )
    assert report["agent"] == inputs.cohort["agent"]
    assert report["grouping_version"] == "dhash-v1"
    assert report["pass"] is True
