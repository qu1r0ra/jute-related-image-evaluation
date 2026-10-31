import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from jute_disease.config import IMAGE_SIZE
from jute_disease.data import PreprocessingSpec, classical_pipeline, create_pipeline


class _AugmentedImageDataset(Dataset):
    def __init__(self, seed: int) -> None:
        self.image = np.random.default_rng(11).integers(
            0, 256, (72, 96, 3), dtype=np.uint8
        )
        self.pipeline = create_pipeline(size=32, is_train=True, seed=seed)

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.pipeline(self.image)


def test_create_pipeline_ml() -> None:
    """ML pipeline should skip normalization and return numpy array."""
    pipeline = create_pipeline(size=256, is_train=False, is_dl=False)
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))

    out = pipeline(img)
    assert isinstance(out, np.ndarray)
    assert out.shape == (256, 256, 3)
    # Check that it's not normalized (values around 0-255)
    assert out.max() > 1.0


def test_create_pipeline_dl() -> None:
    """DL pipeline should include normalization and return torch Tensor."""
    pipeline = create_pipeline(size=256, is_train=False, is_dl=True)
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))

    out = pipeline(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 256, 256)
    # Normalized tensor usually has values around 0 (mean subtraction)
    assert out.min() < 0 or out.max() < 10.0


def test_create_pipeline_high_res() -> None:
    """Pipelines should respect the requested size."""
    pipeline = create_pipeline(size=512, is_train=False, is_dl=True)
    img = Image.fromarray(np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8))

    out = pipeline(img)
    assert out.shape == (3, 512, 512)


def test_pipeline_seed_consistency() -> None:
    """Train pipeline with seed should be consistent for the same size."""
    pipeline1 = create_pipeline(size=256, is_train=True, is_dl=True)
    pipeline2 = create_pipeline(size=256, is_train=True, is_dl=True)

    img = Image.fromarray(np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8))

    out1 = pipeline1(img)
    out2 = pipeline2(img)

    assert torch.allclose(out1, out2)


def test_pipelines_are_independent_and_seed_dependent() -> None:
    """No state is shared between pipelines; the seed alone decides augmentation."""
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (400, 400, 3), np.uint8)
    )
    first = create_pipeline(size=64, is_train=True, is_dl=True, seed=1)(img)
    # Drawing from another pipeline in between must not disturb a fresh one.
    create_pipeline(size=64, is_train=True, is_dl=True, seed=2)(img)
    again = create_pipeline(size=64, is_train=True, is_dl=True, seed=1)(img)
    other = create_pipeline(size=64, is_train=True, is_dl=True, seed=2)(img)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_eval_pipeline_is_deterministic_for_any_seed() -> None:
    img = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (300, 200, 3), np.uint8)
    )
    one = create_pipeline(size=64, is_train=False, is_dl=True, seed=1)(img)
    two = create_pipeline(size=64, is_train=False, is_dl=True, seed=99)(img)
    assert torch.equal(one, two)


def test_training_workers_repeat_for_a_seed_and_advance_independently() -> None:
    def samples(seed: int) -> torch.Tensor:
        loader = DataLoader(
            _AugmentedImageDataset(seed),
            batch_size=None,
            num_workers=2,
            generator=torch.Generator().manual_seed(seed),
        )
        return torch.stack(list(loader))

    first = samples(42)
    assert torch.equal(first, samples(42))
    assert not torch.equal(first, samples(43))


def test_classical_pipeline_is_deterministic_and_unnormalised() -> None:
    img = Image.fromarray(
        np.random.default_rng(4).integers(0, 256, (300, 300, 3), np.uint8)
    )
    out = classical_pipeline(size=IMAGE_SIZE)(img)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, classical_pipeline(size=IMAGE_SIZE)(img))


def test_classical_and_dl_routes_share_uint8_letterbox_padding() -> None:
    image = Image.fromarray(np.full((20, 40, 3), (12, 80, 160), dtype=np.uint8))

    classical = classical_pipeline(size=40)(image)
    deep_learning = create_pipeline(size=40, is_dl=True)(image)
    restored = (
        deep_learning * torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        + torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    )
    restored = (restored.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    expected_padding = np.array((124, 116, 104), dtype=np.uint8)
    assert np.array_equal(classical, restored)
    assert np.array_equal(classical[0], np.tile(expected_padding, (40, 1)))


def test_preprocessing_spec_digest_changes_with_every_field() -> None:
    base = PreprocessingSpec()
    assert base.digest() == PreprocessingSpec().digest()
    for changed in (
        PreprocessingSpec(size=base.size + 1),
        PreprocessingSpec(cache_shortest_side=base.cache_shortest_side + 1),
        PreprocessingSpec(cache_format="webp"),
        PreprocessingSpec(resize="other"),
        PreprocessingSpec(interpolation="other"),
        PreprocessingSpec(padding="other"),
        PreprocessingSpec(normalization="other"),
        PreprocessingSpec(cache_interpolation="other"),
        PreprocessingSpec(revision=base.revision + 1),
    ):
        assert changed.digest() != base.digest()
