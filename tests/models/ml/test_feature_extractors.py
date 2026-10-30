import numpy as np
import pytest
from PIL import Image

from jute_disease.config import IMAGE_SIZE
from jute_disease.models.ml import CraftedFeatureExtractor, RawPixelFeatureExtractor


@pytest.fixture
def sample_numpy_image() -> np.ndarray:
    """Create a dummy numpy image (H, W, C) for testing."""
    return np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a dummy PIL image for testing."""
    arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_raw_pixel_extractor_numpy(sample_numpy_image: np.ndarray) -> None:
    """Test RawPixelFeatureExtractor with numpy input."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    features = extractor.extract(sample_numpy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (IMAGE_SIZE * IMAGE_SIZE * 3,)
    assert features.dtype == np.float32


def test_raw_pixel_extractor_pil(sample_pil_image: Image.Image) -> None:
    """Test RawPixelFeatureExtractor with PIL input."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    features = extractor.extract(sample_pil_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (IMAGE_SIZE * IMAGE_SIZE * 3,)


def test_crafted_extractor_numpy(sample_numpy_image: np.ndarray) -> None:
    """Test CraftedFeatureExtractor with numpy input."""
    extractor = CraftedFeatureExtractor()
    features = extractor.extract(sample_numpy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0
    assert features.dtype == np.float32


def test_crafted_extractor_pil(sample_pil_image: Image.Image) -> None:
    """Test CraftedFeatureExtractor with PIL input."""
    extractor = CraftedFeatureExtractor()
    features = extractor.extract(sample_pil_image)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0


class _Dataset:
    """A tiny dataset with an identity, like ManifestDataset."""

    def __init__(self, identity: dict[str, str] | None, length: int = 2) -> None:
        if identity is not None:
            self.identity = identity
        self.length = length
        self.reads = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        self.reads += 1
        return np.full((8, 8, 3), index, dtype=np.uint8), index % 2


IDENTITY = {
    "manifest": "m1",
    "partition": "train",
    "source": "original",
    "transform": "size=8",
    "deterministic": "true",
}


def test_extract_features_without_identity_never_caches(tmp_path, monkeypatch) -> None:
    import jute_disease.models.ml.features as features_module

    monkeypatch.setattr(features_module, "ML_FEATURES_DIR", tmp_path)
    dataset = _Dataset(None)
    x, y = features_module.extract_features(dataset, RawPixelFeatureExtractor(8))
    assert x.shape == (2, 8 * 8 * 3) and y.tolist() == [0, 1]
    assert not list(tmp_path.iterdir())


def test_feature_cache_hits_and_is_keyed_by_every_input(tmp_path, monkeypatch) -> None:
    import jute_disease.models.ml.features as features_module

    monkeypatch.setattr(features_module, "ML_FEATURES_DIR", tmp_path)
    extract = features_module.extract_features
    extractor = RawPixelFeatureExtractor(8)

    first = _Dataset(IDENTITY)
    extract(first, extractor)
    assert first.reads == 2

    again = _Dataset(IDENTITY)
    extract(again, extractor)
    assert again.reads == 0  # served from the cache

    for changed in (
        _Dataset({**IDENTITY, "manifest": "m2"}),
        _Dataset({**IDENTITY, "transform": "size=16"}),
        _Dataset(IDENTITY, length=3),
    ):
        extract(changed, extractor)
        assert changed.reads == len(changed)  # different inputs recompute
    other = _Dataset(IDENTITY)
    extract(other, RawPixelFeatureExtractor(16))
    assert other.reads == 2  # different extractor settings recompute


def test_feature_cache_ignores_a_stale_sidecar_and_random_transforms(
    tmp_path, monkeypatch
) -> None:
    import json

    import jute_disease.models.ml.features as features_module

    monkeypatch.setattr(features_module, "ML_FEATURES_DIR", tmp_path)
    extract = features_module.extract_features
    extractor = RawPixelFeatureExtractor(8)
    extract(_Dataset(IDENTITY), extractor)

    (sidecar,) = tmp_path.glob("*.json")
    meta = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**meta, "length": 99}))
    stale = _Dataset(IDENTITY)
    extract(stale, extractor)
    assert stale.reads == 2

    sidecar.unlink()
    no_sidecar = _Dataset(IDENTITY)
    extract(no_sidecar, extractor)
    assert no_sidecar.reads == 2

    augmented = _Dataset({**IDENTITY, "deterministic": "false"})
    extract(augmented, extractor)
    extract(augmented, extractor)
    assert augmented.reads == 4  # random augmentation is never cached


def test_feature_cache_uses_an_explicit_directory(tmp_path, monkeypatch) -> None:
    import jute_disease.models.ml.features as features_module

    default = tmp_path / "default"
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(features_module, "ML_FEATURES_DIR", default)
    features_module.extract_features(
        _Dataset(IDENTITY), RawPixelFeatureExtractor(8), cache_dir=explicit
    )
    assert list(explicit.glob("*.json"))
    assert not default.exists()


def test_feature_cache_rejects_a_truncated_or_misshapen_matrix(tmp_path) -> None:
    import numpy as np

    import jute_disease.models.ml.features as features_module

    extractor = RawPixelFeatureExtractor(8)
    features_module.extract_features(_Dataset(IDENTITY), extractor, cache_dir=tmp_path)
    (x_path,) = tmp_path.glob("*_X.npy")

    np.save(x_path, np.zeros((1, 8 * 8 * 3), dtype=np.float32))
    wrong_rows = _Dataset(IDENTITY)
    x, _ = features_module.extract_features(wrong_rows, extractor, cache_dir=tmp_path)
    assert wrong_rows.reads == 2 and x.shape[0] == 2

    x_path.write_bytes(b"not an array")
    corrupt = _Dataset(IDENTITY)
    features_module.extract_features(corrupt, extractor, cache_dir=tmp_path)
    assert corrupt.reads == 2

    again = _Dataset(IDENTITY)
    features_module.extract_features(again, extractor, cache_dir=tmp_path)
    assert again.reads == 0  # the rewritten cache is whole


def test_feature_cache_publishes_the_sidecar_last(tmp_path, monkeypatch) -> None:
    import numpy as np

    import jute_disease.models.ml.features as features_module

    order: list[str] = []
    real_save = np.save

    def spy(path, array, *args, **kwargs):
        order.append(f"npy:{'X' if '_X' in str(path) else 'y'}")
        real_save(path, array, *args, **kwargs)

    monkeypatch.setattr(features_module.np, "save", spy)
    features_module.extract_features(
        _Dataset(IDENTITY), RawPixelFeatureExtractor(8), cache_dir=tmp_path
    )
    assert order == ["npy:X", "npy:y"]
    assert not list(tmp_path.glob("*.part*"))
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_an_interrupted_write_leaves_no_valid_cache(tmp_path, monkeypatch) -> None:
    import jute_disease.models.ml.features as features_module

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(features_module.os, "replace", boom)
    with pytest.raises(OSError):
        features_module.extract_features(
            _Dataset(IDENTITY), RawPixelFeatureExtractor(8), cache_dir=tmp_path
        )
    assert not list(tmp_path.glob("*.json"))
    clean = _Dataset(IDENTITY)
    monkeypatch.undo()
    features_module.extract_features(
        clean, RawPixelFeatureExtractor(8), cache_dir=tmp_path
    )
    assert clean.reads == 2
