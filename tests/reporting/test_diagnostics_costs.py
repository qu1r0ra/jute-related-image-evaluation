"""Receipt-only compute cost reporting fixtures; no model work is run."""

from pathlib import Path

from jute_disease.reporting import compute_training_costs
from jute_disease.training import RunStore


def _completed_receipt(
    store: RunStore,
    fit_id: str,
    fit: dict[str, object],
    *,
    a100: bool,
) -> None:
    active = store.start(fit_id, f"identity-{fit_id}")
    artifact = active.attempt_dir / "synthetic.bin"
    artifact.write_bytes(b"synthetic receipt fixture")
    metadata = (
        {
            "runtime": {
                "cuda_available": True,
                "device": {"name": "NVIDIA A100 synthetic fixture"},
            }
        }
        if a100
        else {}
    )
    active.complete({"fit": fit}, {"fixture": artifact}, metadata=metadata)


def test_diagnostics_costs_separate_gpu_search_and_classical_search_receipts(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    store = RunStore(runs)
    _completed_receipt(
        store,
        "cnn/tune/mobilenet_v2_direct/pool-A",
        {"kind": "target", "phase": "tune", "upstream": None},
        a100=True,
    )
    _completed_receipt(
        store,
        "classical/tune/A/gnb-raw",
        {"kind": "classical", "phase": "tune"},
        a100=False,
    )

    report = compute_training_costs(runs)

    search = next(row for row in report["phases"] if row["phase"] == "search")
    assert search["fit_count"] == 1
    assert search["attempt_count"] == 1
    assert search["a100_gpu_hours"] >= 0
    assert report["stages"][0]["stage"] == "core"
    assert report["classical_search"]["fit_count"] == 1
    assert report["classical_search"]["attempt_count"] == 1
    assert report["classical_search"]["attempt_seconds"] >= 0
    assert report["classical_search"]["configurations_per_fit"]["gnb"] == 9
    assert report["classical_search"]["receipts"][0]["sha256"]
