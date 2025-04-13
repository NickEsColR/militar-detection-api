import torch
import torch.nn as nn
from torch import Tensor


class FeatureExtractor(nn.Module):
    """
    Extracts features using the convolutional part of a model.
    Allows freezing layers up to a given index.

    Args:
        model (nn.Module): The model to extract features from.
        freeze_until (int, optional): Index of the last layer to freeze. Defaults to -1 (no freezing).
    """

    def __init__(self, model: nn.Module, freeze_until: int = -1):
        super().__init__()
        all_features = list(model.features)
        if freeze_until >= 0:
            self.frozen = nn.Sequential(*all_features[: freeze_until + 1])
            self.trainable = nn.Sequential(*all_features[freeze_until + 1 :])
            for param in self.frozen.parameters():
                param.requires_grad_(False)
        else:
            self.features = nn.Sequential(*all_features)
        # Check if the pretrained model has an avgpool layer
        if hasattr(model, "avgpool"):
            self.pooling = model.avgpool
        else:
            # If not, use PyTorch's AdaptiveAvgPool2d
            self.pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.freeze_until = freeze_until

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Extracted features.
        """
        if hasattr(self, "frozen"):
            x = self.frozen(x)
            x = self.trainable(x)
        else:
            x = self.features(x)
        return self.pooling(x)
