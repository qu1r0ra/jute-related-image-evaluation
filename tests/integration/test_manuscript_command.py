"""Manuscript fixture exports remain tied to synthetic frozen evidence."""

import json
import subprocess
import sys
from pathlib import Path

from jute_disease.reporting.manuscript import _validate_figure_sources


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_fixture_freezes_exports_rebuilds_and_checks_macros(tmp_path: Path) -> None:
    output = tmp_path / "manuscript-fixture"
    result = _run("manuscript", "fixture", "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["check"]["pass"] is True
    assert report["check"]["checked_macros"] > 0
    assert report["rebuild"]["verified_images"] > 0

    export_dir = output / "export"
    metadata = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    macros = json.loads((export_dir / "macros.json").read_text(encoding="utf-8"))
    audit = json.loads((export_dir / "claim-audit.json").read_text(encoding="utf-8"))
    assert metadata["command"] == "jute-disease manuscript fixture"
    assert metadata["configuration"]["purpose"] == "synthetic fixture only"
    assert metadata["artifacts"]["manifest"]["manifest_digest"]
    assert len(macros) == 1 and report["export"]["export_id"] in macros
    assert (
        len(macros[report["export"]["export_id"]]) == report["check"]["checked_macros"]
    )
    assert audit["status"] == "pending"
    assert audit["owner_spot_check"]["status"] == "pending"
    assert audit["owner_signoff"]["status"] == "pending"
    assert audit["claim_lists"]["may_report"]
    assert audit["claim_lists"]["may_not_claim"]

    rebuild_script = Path(__file__).parents[2] / "scripts" / "rebuild_splits.py"
    rebuilt = subprocess.run(
        [
            sys.executable,
            str(rebuild_script),
            "--kit-manifest",
            str(export_dir / "rebuild-kit.json"),
            "--image-root",
            str(output / "images"),
            "--output",
            str(tmp_path / "script-rebuilt-splits.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert (
        json.loads(rebuilt.stdout)["split_digest"] == report["rebuild"]["split_digest"]
    )

    macros[report["export"]["export_id"]][
        next(iter(macros[report["export"]["export_id"]]))
    ] += 1
    (export_dir / "macros.json").write_text(json.dumps(macros), encoding="utf-8")
    mismatch = _run("manuscript", "check", str(export_dir))
    assert mismatch.returncode != 0
    assert "macro JSON values" in mismatch.stderr


def test_figure_sources_require_allowed_license_and_attribution() -> None:
    _validate_figure_sources(
        [
            {
                "source": "Example",
                "url": "https://example.invalid/image",
                "license": "CC BY",
                "attribution": "Author",
            }
        ]
    )
    try:
        _validate_figure_sources(
            [
                {
                    "source": "Example",
                    "url": "https://example.invalid/image",
                    "license": "All rights reserved",
                    "attribution": "Author",
                }
            ]
        )
    except ValueError as error:
        assert "unsupported license" in str(error)
    else:
        raise AssertionError("unsupported figure license should be rejected")
