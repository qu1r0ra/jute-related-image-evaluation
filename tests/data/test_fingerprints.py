"""Fingerprints: byte/pixel identity and dihedral perceptual hashing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.data import fingerprint_image, phash_distance
from jute_disease.data.fingerprints import PHASH_VARIANTS, perceptual_hashes


def _field(seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)


def test_perceptual_hashes_has_all_dihedral_variants() -> None:
    hashes = perceptual_hashes(_field(1))
    assert len(hashes) == PHASH_VARIANTS
    assert all(0 <= value < 2**64 for value in hashes)


def test_rotated_copy_is_a_close_candidate() -> None:
    original = _field(2)
    turned = original.transpose(Image.Transpose.ROTATE_90)
    assert phash_distance(perceptual_hashes(original), perceptual_hashes(turned)) <= 6


def test_unrelated_images_are_far_apart() -> None:
    distance = phash_distance(
        perceptual_hashes(_field(3)), perceptual_hashes(_field(4))
    )
    assert distance > 6


def test_fingerprint_is_deterministic_and_matches_content(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    _field(5).save(path)
    first = fingerprint_image(path)
    assert first == fingerprint_image(path)
    assert (first.width, first.height) == (64, 64)
    assert first.bytes == path.stat().st_size


def test_byte_copy_shares_both_hashes_but_reencoding_changes_bytes(
    tmp_path: Path,
) -> None:
    original = tmp_path / "a.png"
    _field(6).save(original)
    copy = tmp_path / "b.png"
    copy.write_bytes(original.read_bytes())
    reencoded = tmp_path / "c.png"
    Image.open(original).save(reencoded, compress_level=0)
    a, b, c = (fingerprint_image(p) for p in (original, copy, reencoded))
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256
    assert a.pixel_sha256 == c.pixel_sha256


def test_undecodable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        fingerprint_image(path)
