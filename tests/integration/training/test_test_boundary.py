"""Training never builds or opens the test partition."""

import pytest

from jute_disease.data import DataModule, HeldOutPartitionError

from .conftest import Inputs


def _module(inputs: Inputs, **kwargs: object) -> DataModule:
    return DataModule(
        manifest_dir=inputs.target_dir,
        image_root=inputs.image_root,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        **kwargs,
    )


def test_default_setup_builds_only_training_partitions(inputs: Inputs) -> None:
    module = _module(inputs)
    module.setup()
    assert module.jute_train is not None
    assert module.jute_val is not None
    assert module.jute_test is None


@pytest.mark.parametrize("stage", ["test", "predict"])
def test_test_and_predict_stages_are_denied_by_default(
    inputs: Inputs, stage: str
) -> None:
    module = _module(inputs)
    with pytest.raises(HeldOutPartitionError):
        module.setup(stage)
    assert module.jute_test is None
    assert module.jute_predict is None


def test_evaluation_must_opt_in_to_the_test_partition(inputs: Inputs) -> None:
    module = _module(inputs, allow_test=True)
    module.setup("test")
    assert module.jute_test is not None
