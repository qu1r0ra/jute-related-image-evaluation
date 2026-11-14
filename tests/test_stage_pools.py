"""Pool staging validates the archives before it writes any image."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "stage_pools.py"

POOL_A_MEMBERS = {
    "kaggle/Cescospora Leaf Spot/a.jpg": b"a1",
    "kaggle/Golden Mosaic/b.jpg": b"b1",
    "kaggle/Healthy Leaf/c.jpg": b"c1",
}
POOL_B_MEMBERS = {
    "Jute Diesease Dataset/Dieback-300/a.jpg": b"a2",
    "Jute Diesease Dataset/Fresh-280/b.jpg": b"b2",
    "Jute Diesease Dataset/Holed-300/c.jpg": b"c2",
    "Jute Diesease Dataset/Mosaic-240/d.jpg": b"d2",
    "Jute Diesease Dataset/Stem Soft Rot-270/e.jpg": b"e2",
}
POOL_C_MEMBERS = {
    "Mosaic/a.jpg": b"a3",
    "Soft_Rot/b.jpg": b"b3",
    "Stem_Rot/c.jpg": b"c3",
    "Healthy_19/CC_45/d.jpg": b"d3",
}


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def _archives(root: Path, pool_a: dict[str, bytes] | None = None) -> Path:
    archives = root / "archives"
    _zip(archives / "J01" / "kaggle.zip", POOL_A_MEMBERS if pool_a is None else pool_a)
    _zip(archives / "J02" / "Jute Disease Dataset.zip", POOL_B_MEMBERS)
    _zip(archives / "J22" / "kaggle.zip", POOL_C_MEMBERS)
    return archives


def _stage(archives: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archives",
            str(archives),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_staging_records_the_archive_sha256_in_each_receipt(tmp_path: Path) -> None:
    archives = _archives(tmp_path)
    output = tmp_path / "pools"

    result = _stage(archives, output)

    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "A" / "stage-receipt.json").read_text("utf-8"))
    archive = archives / "J01" / "kaggle.zip"
    assert receipt["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert receipt["archive"] == "kaggle.zip"
    assert (output / "A" / "Healthy" / "c.jpg").read_bytes() == b"c1"
    assert (output / "C" / "Healthy" / "CC_45__d.jpg").read_bytes() == b"d3"
    assert not list(output.glob(".*.staging"))


def test_staging_rejects_paths_that_differ_only_by_case(tmp_path: Path) -> None:
    members = POOL_A_MEMBERS | {"kaggle/Healthy Leaf/C.JPG": b"other"}
    archives = _archives(tmp_path, members)
    output = tmp_path / "pools"

    result = _stage(archives, output)

    assert result.returncode != 0
    assert "case-insensitively" in result.stderr
    assert not (output / "A").exists()
    assert not list(output.glob(".*.staging"))


def test_staging_rejects_an_archive_that_yields_no_images(tmp_path: Path) -> None:
    archives = _archives(tmp_path, {"kaggle/unrelated/readme.txt": b"text"})
    output = tmp_path / "pools"

    result = _stage(archives, output)

    assert result.returncode != 0
    assert "staged no images" in result.stderr
    assert not (output / "A").exists()


def test_staging_rejects_an_empty_image(tmp_path: Path) -> None:
    members = POOL_A_MEMBERS | {"kaggle/Healthy Leaf/empty.jpg": b""}
    archives = _archives(tmp_path, members)

    result = _stage(archives, tmp_path / "pools")

    assert result.returncode != 0
    assert "empty image" in result.stderr


def test_staging_rejects_a_pool_missing_a_label(tmp_path: Path) -> None:
    members = {
        name: data for name, data in POOL_A_MEMBERS.items() if "Golden" not in name
    }
    archives = _archives(tmp_path, members)

    result = _stage(archives, tmp_path / "pools")

    assert result.returncode != 0
    assert "Golden Mosaic" in result.stderr


def test_staging_refuses_to_mix_into_an_existing_pool(tmp_path: Path) -> None:
    archives = _archives(tmp_path)
    output = tmp_path / "pools"
    assert _stage(archives, output).returncode == 0

    again = _stage(archives, output)

    assert again.returncode != 0
    assert "already staged" in again.stderr


@pytest.mark.parametrize("pool", ["B", "C"])
def test_staging_validates_every_pool(tmp_path: Path, pool: str) -> None:
    archives = _archives(tmp_path)
    broken = {
        "B": archives / "J02" / "Jute Disease Dataset.zip",
        "C": archives / "J22" / "kaggle.zip",
    }[pool]
    _zip(broken, {"nothing/here.txt": b"x"})

    result = _stage(archives, tmp_path / "pools")

    assert result.returncode != 0
    assert f"pool {pool}" in result.stderr
