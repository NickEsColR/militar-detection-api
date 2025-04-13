import torch
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from feature_extractor import FeatureExtractor


def get_pretrained_model(freeze_until):
    """
    Loads a pre-trained SqueezeNet1_0 model, freezes its convolutional layers,
    and replaces the classifier with a new fully connected layer for the
    specified number of output classes.

    Args:
        freeze_until (int): Index of the last convolutional layer to freeze.

    Returns:
        torch.nn.Module: The modified SqueezeNet model.
    """
    model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)

    return FeatureExtractor(model, freeze_until)
