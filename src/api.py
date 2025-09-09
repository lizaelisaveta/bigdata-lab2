import os
import logging
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cassandra_client import CassandraClient
from src.config import IMG_WIDTH, IMG_HEIGHT, MODEL_PATH, API_HOST, API_PORT, MAX_FILE_SIZE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Cats vs Dogs API")
model = None
client = None


@app.on_event("startup")
def startup_event():
    global model, client
    try:
        client = CassandraClient()
        if os.path.exists(MODEL_PATH):
            logger.info("Loading model...")
            model = load_model(MODEL_PATH)
            logger.info("✅ Model loaded")
        else:
            logger.warning(f"⚠️ Model file {MODEL_PATH} not found. Train the model first!")
            model = None
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


@app.get("/active")
async def active():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG/PNG files are supported")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        img = Image.open(file.file).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        label = "dog" if class_idx == 1 else "cat"

        prediction_id = uuid.uuid4()
        client.insert_prediction(prediction_id, label, confidence)

        return JSONResponse(
            content={
                "id": str(prediction_id),
                "predicted_label": label,
                "confidence": confidence,
            }
        )   

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
