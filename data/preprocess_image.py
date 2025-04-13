from PIL import Image
import torch

from ..config import TARGET_SIZE
from pipeline import get_transforms


pipeline = get_transforms() 

def preprocess_image(image_data: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize(TARGET_SIZE)

        image_tensor = pipeline(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format or preprocessing error: {e}",
        )
