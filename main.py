from fastapi import FastAPI, HTTPException, status
import base64
import sys

sys.path.append("./model")
sys.path.append("./data")

from data.preprocess_image import preprocess_image
# MultiTaskModel and FeatureExtractor imports are required to load the model
from model import load_model, MultiTaskModel
from feature_extractor import FeatureExtractor


model = load_model()

app = FastAPI()

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(image_base64: str):
    try:
        image_bytes = base64.b64decode(image_base64)
        image_tensor = preprocess_image(image_bytes)

        preds = model(image_tensor)

        return preds

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e}",
        )
