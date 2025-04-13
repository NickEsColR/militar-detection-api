import numpy as np
from typing import Dict, Any, Tuple, Callable
import torch
import torchvision

from config import MEANS, STDS

class ToTensor(object):
    """
    Converts a NumPy array image to a PyTorch tensor.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the transformation to the sample.

        Args:
            sample (Dict[str, Any]): A dictionary containing the image.

        Returns:
            Dict[str, Any]: A dictionary containing the transformed image.
        """
        image = sample["image"]
        image = image.transpose((2, 0, 1))
        sample["image"] = torch.from_numpy(image).float()
        return sample


class Normalizer(object):
    """
    Normalizes an image using provided means and standard deviations.

    Args:
        means (np.ndarray): Means for each channel.
        stds (np.ndarray): Standard deviations for each channel.
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self.means = means
        self.stds = stds

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the normalization to the sample.

        Args:
            sample (Dict[str, Any]): A dictionary containing the image.

        Returns:
            Dict[str, Any]: A dictionary containing the normalized image.
        """
        image = sample["image"]
        for channel in range(image.shape[0]):
            image[channel] = (image[channel] - self.means[channel]) / self.stds[channel]
        sample["image"] = image
        return sample


def get_transforms() -> Tuple[Callable, Callable]:
    """
    Builds and returns the transformation pipelines.

    Returns:
        Tuple[Callable, Callable]: A tuple containing the training and evaluation transforms.
    """
    return torchvision.transforms.Compose([ToTensor(), Normalizer(MEANS, STDS)])
