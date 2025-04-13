import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from torch import Tensor
from functools import reduce
from torchvision.models import vgg16, VGG16_Weights

from transfer import get_pretrained_model
from config import FREEZE_UNITL, TARGET_SIZE,OBJ2ID

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class MultiTaskModel(nn.Module):
    """
    Multi-task model for classification and bounding box regression.
    Uses VGG16 with pretrained weights as the backbone if not provided.

    Args:
        input_shape (Tuple[int, int, int]): Input shape of the images.
        n_classes (int): Number of classes.
        backbone (Optional[nn.Module], optional): Backbone model to use. Defaults to None.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_shape = input_shape
        if backbone is None:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            backbone = FeatureExtractor(vgg, freeze_until=15)
        self.backbone = backbone.to(DEVICE)
        dummy_input = torch.rand(1, *input_shape).to(DEVICE)
        backbone_out_shape = self.backbone(dummy_input).shape
        flattened_features = reduce(lambda x, y: x * y, backbone_out_shape[1:])
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_shape[1], out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone_out_shape[1],
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass of the multi-task model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Dict[str, Tensor]: A dictionary containing class logits and bounding box predictions.
        """
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_pred = torch.sigmoid(self.reg_head(features))
        return {"class_id": cls_logits, "bbox": bbox_pred}

def load_model():
    backbone = get_pretrained_model(freeze_until=FREEZE_UNITL)
    model = MultiTaskModel(input_shape=(3, TARGET_SIZE[1], TARGET_SIZE[0]), n_classes=len(OBJ2ID), backbone=backbone)
    model.load_state_dict(torch.load("./model/weights.pth", map_location=torch.device(DEVICE), weights_only=True))
    # torch.serialization.add_safe_globals([MultiTaskModel])
    # model = torch.load(
    #     "./model/model.pth", map_location=torch.device("cpu"), weights_only=False
    # )
    model.eval()
    return model