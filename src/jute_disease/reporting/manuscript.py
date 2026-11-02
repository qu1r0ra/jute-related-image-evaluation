"""Reproducible manuscript exports and their evidence checks."""

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from jute_disease.data import load_manifest, rebuild_partition_assignments
from jute_disease.data.manifest import PARTITIONS

CLAIM_LIST_URL = (
    "https://github.com/qu1r0ra/jute-related-image-evaluation/issues/108"
    "#issuecomment-6545452441"
)
ALLOWED_FIGURE_LICENSES = {"CC0", "CC BY", "CC BY-SA"}
_EXPORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _label_key(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def _support_values(manifest: Any) -> dict[str, int]:
    support = manifest.support()
    return {
        f"{partition}_{_label_key(label)}_n": support[partition].get(label, 0)
        for partition in PARTITIONS
        for label in manifest.labels
    }


def _macro_tex(export_id: str, values: dict[str, int]) -> str:
    rows = [
        f"\\expandafter\\def\\csname juteExport@{export_id}@{key}\\endcsname{{{value}}}"
        for key, value in sorted(values.items())
    ]
    return "\n".join(rows) + "\n"


def _validate_figure_sources(sources: list[dict[str, Any]]) -> None:
    for index, source in enumerate(sources):
        if source.get("license") not in ALLOWED_FIGURE_LICENSES:
            raise ValueError(
                f"figure source {index} has unsupported license; allowed: "
                f"{', '.join(sorted(ALLOWED_FIGURE_LICENSES))}"
            )
        for field in ("source", "url", "attribution"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"figure source {index} requires {field}")
        if "path" in source and not isinstance(source["path"], str):
            raise ValueError(f"figure source {index} path must be a string")


def _write_figure(path: Path, manifest: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = manifest.labels
    support = manifest.support()
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.35), 5.5))
    for index, partition in enumerate(PARTITIONS):
        values = [support[partition].get(label, 0) for label in labels]
        axis.bar(x + (index - 1) * width, values, width, label=partition)
        for offset, value in zip(x + (index - 1) * width, values, strict=True):
            axis.annotate(
                str(value),
                (offset, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_yticks([])
    axis.set_ylabel("Images")
    axis.set_title("Frozen image support by partition and class")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "jute-disease"})
    plt.close(figure)


def export_manuscript(
    freeze_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    source_url: str,
    source_version: str,
    figure_sources_path: Path | None = None,
    export_id: str | None = None,
    command: str = "jute-disease manuscript export",
) -> dict[str, Any]:
    """Write a table, figure, macro files, rebuild kit and audit template.

    The source freeze must already contain a matching manifest and receipt. An
    existing non-empty output directory is never overwritten.
    """
    freeze_dir = Path(freeze_dir)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    manifest = load_manifest(freeze_dir)
    receipt_path = freeze_dir / "receipt.json"
    manifest_path = freeze_dir / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("export configuration must be a JSON object")
    config_bytes = _canonical_json(config) + b"\n"
    if figure_sources_path is None:
        figure_sources: list[dict[str, Any]] = []
    else:
        figure_sources = json.loads(
            Path(figure_sources_path).read_text(encoding="utf-8")
        )
        if not isinstance(figure_sources, list):
            raise ValueError("figure source credits must be a JSON array")
    _validate_figure_sources(figure_sources)

    identity = {
        "manifest_digest": manifest.digest(),
        "configuration_sha256": _sha256_bytes(config_bytes),
    }
    selected_id = export_id or f"exp-{_sha256_bytes(_canonical_json(identity))[:12]}"
    if not _EXPORT_ID.fullmatch(selected_id):
        raise ValueError(
            "export ID may contain letters, digits, dot, underscore and dash"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(
            f"Refusing to overwrite non-empty export directory: {output_dir}"
        )

    values = _support_values(manifest)
    macros = {selected_id: values}
    (output_dir / "configuration.json").write_bytes(config_bytes)
    _write_json(output_dir / "macros.json", macros)
    (output_dir / "macros.tex").write_text(
        _macro_tex(selected_id, values), encoding="utf-8", newline="\n"
    )
    with (output_dir / "support.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("partition", "label", "count"))
        writer.writeheader()
        support = manifest.support()
        for partition in PARTITIONS:
            for label in manifest.labels:
                writer.writerow(
                    {
                        "partition": partition,
                        "label": label,
                        "count": support[partition].get(label, 0),
                    }
                )
    _write_figure(output_dir / "support.png", manifest)
    _write_json(output_dir / "figure-sources.json", figure_sources)

    kit_entries = [
        {
            "source": entry.source,
            "url": source_url,
            "version": source_version,
            "path": entry.path,
            "sha256": entry.image_id,
            "family_id": entry.group,
            "split": entry.partition,
            "label": entry.label,
            "image_id": entry.image_id,
        }
        for entry in manifest.entries
    ]
    kit = {
        "version": 1,
        "seed": manifest.seed,
        "fractions": dict(manifest.fractions),
        "entries": kit_entries,
    }
    _write_json(output_dir / "rebuild-kit.json", kit)

    manifest_ref_text = Path(
        os.path.relpath(manifest_path.resolve(), output_dir.resolve())
    ).as_posix()
    receipt_ref_text = Path(
        os.path.relpath(receipt_path.resolve(), output_dir.resolve())
    ).as_posix()

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "export_id": selected_id,
        "command": command,
        "configuration": config,
        "configuration_file": "configuration.json",
        "configuration_sha256": _sha256_file(output_dir / "configuration.json"),
        "artifacts": {
            "manifest": {
                "path": manifest_ref_text,
                "sha256": _sha256_file(manifest_path),
                "manifest_digest": manifest.digest(),
            },
            "receipt": {
                "path": receipt_ref_text,
                "sha256": _sha256_file(receipt_path),
            },
        },
        "outputs": {
            name: {"path": name, "sha256": _sha256_file(output_dir / name)}
            for name in (
                "configuration.json",
                "macros.json",
                "macros.tex",
                "support.csv",
                "support.png",
                "figure-sources.json",
                "rebuild-kit.json",
            )
        },
    }
    _write_json(output_dir / "export.json", metadata)
    audit = {
        "issue": 108,
        "claim_list_url": CLAIM_LIST_URL,
        "status": "pending",
        "auditor_requirements": (
            "Fresh context; reviewer provider differs from the writing agent; "
            "review only manuscript, macros, claim lists and artifacts."
        ),
        "claim_lists": {
            "may_report": [
                "The grouping rule, its threshold and the leakage risk that remains.",
                "Macro-F1 differences between policies for this pool and this rule, "
                "with hierarchical bootstrap intervals.",
                "Route rankings, conditional on MobileNetV2, its fixed source "
                "checkpoints and this pool.",
                "Grad-CAM as a prespecified qualitative diagnostic.",
                "Descriptive t-SNE/UMAP plots at fixed, recorded settings.",
                "Background probes, bounded to how much label information the "
                "background carries in this pool.",
            ],
            "may_not_claim": [
                "Leakage-free or zero leakage; use 'we did not identify' instead.",
                "Plant- or session-independent splits.",
                "Field, deployment or new-acquisition generalization.",
                "Rankings that extend beyond MobileNetV2.",
                "The model attends to the lesion.",
                "Meaning from t-SNE/UMAP cluster sizes or distances, or their use "
                "as evidence of performance or leakage.",
                "Family-disjoint scores presented as the true field estimate.",
            ],
        },
        "macro_rows": [
            {"key": key, "value": value, "status": "pending", "evidence": ""}
            for key, value in sorted(values.items())
        ],
        "claim_rows": [],
        "unflagged_numeric_statements": [],
        "blocking_rule": "Any fail or unsupported row blocks submission; no waiver.",
        "owner_spot_check": {
            "status": "pending",
            "fraction": 0.1,
            "selection_record": "",
        },
        "owner_signoff": {"status": "pending", "signed_by": "", "signed_at": ""},
        "re_audit_rule": (
            "Re-audit changed numbers and claims after every manuscript change."
        ),
        "allowed_claims_reference": CLAIM_LIST_URL,
        "prohibited_claims_reference": CLAIM_LIST_URL,
    }
    _write_json(output_dir / "claim-audit.json", audit)
    return {"export_id": selected_id, "output_dir": str(output_dir), "macros": macros}


def rebuild_kit_splits(
    kit_path: Path, image_root: Path, output_path: Path
) -> dict[str, Any]:
    """Verify source bytes and reproduce the kit's deterministic split map."""
    kit = json.loads(Path(kit_path).read_text(encoding="utf-8"))
    entries = kit.get("entries")
    if kit.get("version") != 1 or not isinstance(entries, list):
        raise ValueError("unsupported or malformed rebuild kit")
    image_root = Path(image_root)
    for entry in entries:
        path = image_root / Path(entry["path"])
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"source hash mismatch for {entry['path']}")
    assignments = rebuild_partition_assignments(entries, kit["seed"], kit["fractions"])
    expected = {entry["image_id"]: entry["split"] for entry in entries}
    if assignments != expected:
        mismatched = next(
            key for key in sorted(expected) if assignments.get(key) != expected[key]
        )
        raise ValueError(f"rebuilt split differs for image {mismatched}")
    report = {
        "version": 1,
        "source_manifest": str(Path(kit_path)),
        "verified_images": len(entries),
        "assignments": dict(sorted(assignments.items())),
        "split_digest": _sha256_bytes(_canonical_json(sorted(assignments.items()))),
    }
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def validate_export(output_dir: Path) -> dict[str, Any]:
    """Check macros, table, rebuild kit and outputs against frozen artifacts."""
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "export.json").read_text(encoding="utf-8"))
    artifacts = metadata["artifacts"]
    manifest_path = (output_dir / artifacts["manifest"]["path"]).resolve()
    receipt_path = (output_dir / artifacts["receipt"]["path"]).resolve()
    if _sha256_file(manifest_path) != artifacts["manifest"]["sha256"]:
        raise ValueError("authoritative manifest bytes changed after export")
    if _sha256_file(receipt_path) != artifacts["receipt"]["sha256"]:
        raise ValueError("authoritative receipt bytes changed after export")
    manifest = load_manifest(manifest_path.parent)
    if manifest.digest() != artifacts["manifest"]["manifest_digest"]:
        raise ValueError("export lineage manifest digest mismatch")
    if (
        _sha256_file(output_dir / "configuration.json")
        != metadata["configuration_sha256"]
    ):
        raise ValueError("export configuration changed after export")

    export_id = metadata["export_id"]
    expected_values = _support_values(manifest)
    macros = json.loads((output_dir / "macros.json").read_text(encoding="utf-8"))
    if macros != {export_id: expected_values}:
        raise ValueError("macro JSON values do not match the authoritative manifest")
    if (output_dir / "macros.tex").read_text(encoding="utf-8") != _macro_tex(
        export_id, expected_values
    ):
        raise ValueError("TeX macros do not match the authoritative manifest")

    expected_csv = ["partition,label,count"]
    support = manifest.support()
    expected_csv.extend(
        f"{partition},{label},{support[partition].get(label, 0)}"
        for partition in PARTITIONS
        for label in manifest.labels
    )
    csv_text = (
        (output_dir / "support.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
    )
    if csv_text != "\n".join(expected_csv) + "\n":
        raise ValueError(
            "exported support table does not match the authoritative manifest"
        )
    kit = json.loads((output_dir / "rebuild-kit.json").read_text(encoding="utf-8"))
    expected_kit = [
        {
            "source": entry.source,
            "url": row["url"],
            "version": row["version"],
            "path": entry.path,
            "sha256": entry.image_id,
            "family_id": entry.group,
            "split": entry.partition,
            "label": entry.label,
            "image_id": entry.image_id,
        }
        for entry, row in zip(manifest.entries, kit["entries"], strict=True)
    ]
    if kit.get("entries") != expected_kit:
        raise ValueError("rebuild kit entries do not match the authoritative manifest")
    _validate_figure_sources(
        json.loads((output_dir / "figure-sources.json").read_text(encoding="utf-8"))
    )
    for name, record in metadata["outputs"].items():
        if _sha256_file(output_dir / name) != record["sha256"]:
            raise ValueError(f"export output changed after export: {name}")
    return {
        "export_id": export_id,
        "checked_macros": len(expected_values),
        "pass": True,
    }


def run_manuscript_fixture(output_dir: Path) -> dict[str, Any]:
    """Exercise synthetic data, freeze, export, rebuild and mechanical checking."""
    from jute_disease.config import CURATION_RULES_PATH
    from jute_disease.data import make_synthetic_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(
            f"Refusing to overwrite non-empty fixture directory: {output_dir}"
        )
    image_root = output_dir / "images"
    make_synthetic_data(image_root, per_class=20, seed=42)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    rules_path = output_dir / "fixture-rules.json"
    reviews_path = output_dir / "fixture-reviews.json"
    _write_json(rules_path, rules)
    _write_json(reviews_path, {"version": 1, "decisions": []})
    freeze_dir = output_dir / "freeze" / "target"
    freeze_command = [
        sys.executable,
        "-m",
        "jute_disease.cli",
        "freeze",
        "target",
        "--image-root",
        str(image_root),
        "--rules",
        str(rules_path),
        "--reviews",
        str(reviews_path),
        "--output",
        str(output_dir / "freeze"),
        "--seed",
        "42",
        "--workers",
        "2",
    ]
    frozen = subprocess.run(
        freeze_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if frozen.returncode:
        raise ValueError(f"fixture freeze failed: {frozen.stderr.strip()}")
    config = {
        "purpose": "synthetic fixture only",
        "freeze_command": "jute-disease freeze target",
        "seed": 42,
        "fractions": dict(load_manifest(freeze_dir).fractions),
        "rules_sha256": _sha256_file(rules_path),
        "reviews_sha256": _sha256_file(reviews_path),
    }
    config_path = output_dir / "fixture-export-config.json"
    _write_json(config_path, config)
    export_dir = output_dir / "export"
    export = export_manuscript(
        freeze_dir,
        config_path,
        export_dir,
        source_url="synthetic://local-fixture",
        source_version="synthetic-1",
        command="jute-disease manuscript fixture",
    )
    rebuild = rebuild_kit_splits(
        export_dir / "rebuild-kit.json",
        image_root,
        output_dir / "rebuilt-splits.json",
    )
    checked = validate_export(export_dir)
    if rebuild["verified_images"] != len(load_manifest(freeze_dir).entries):
        raise ValueError("fixture rebuild kit omitted frozen images")
    return {"export": export, "rebuild": rebuild, "check": checked}
