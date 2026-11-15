from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int, workers: bool = True) -> None:
    from lightning.pytorch import seed_everything as lightning_seed_everything

    lightning_seed_everything(seed, workers=workers)
    logger.info(f"Random seed set to: {seed}")
