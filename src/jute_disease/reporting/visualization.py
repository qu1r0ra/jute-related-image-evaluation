import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from jute_disease.config import IMAGE_SIZE


def denormalize(
    img_tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    image = img_tensor.permute(1, 2, 0).cpu().numpy()
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def visualize_augmentations(
    dataset: Dataset,
    num_samples: int = 6,
    num_augmentations: int = 7,
    figsize: tuple[int, int] = (15, 15),
) -> None:
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, figsize=figsize)
    plt.tight_layout()

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        for j in range(num_augmentations + 1):
            ax = axes[i, j]
            ax.axis("off")

            # Original image
            if j == 0:
                img_path = dataset.samples[idx][0]
                image = (
                    Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                )
                ax.imshow(image)
                ax.set_title(
                    f"Original\n(Class: {dataset.classes[dataset.samples[idx][1]]})"
                )
            # Augmented image
            else:
                image, label = dataset[idx]
                ax.imshow(denormalize(image))

    plt.show()
