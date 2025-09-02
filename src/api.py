import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging
import os
from contextlib import asynccontextmanager

from src.config import app_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_PATH = app_config['paths']['model_path']
IMG_HEIGHT = int(app_config['model']['input_height'])
IMG_WIDTH = int(app_config['model']['input_width'])
MAX_FILE_SIZE = int(app_config['api']['max_file_size'])
API_PORT = int(app_config['api']['port'])
API_HOST = app_config['api']['host']

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
        else:
            logger.info(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            logger.info(f"Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
    yield  


app = FastAPI(
    title="Dogs vs Cats Classifier API",
    description="API для классификации изображений собак и кошек",
    version="1.0.0",
    lifespan=lifespan
)


CLASS_NAMES = ["Cat", "Dog"]


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")


@app.get("/active")
def activity_check():
    return {
        "status": "active" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "image_size": f"{IMG_WIDTH}x{IMG_HEIGHT}"
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check if model file exists.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE} bytes)")
        
        processed_image = preprocess_image(contents)
        prediction = model.predict(processed_image, verbose=0)
        
        predicted_class = CLASS_NAMES[int(prediction[0][0] > 0.5)]
        confidence_score = float(prediction[0][0])
        confidence = confidence_score if predicted_class == "Dog" else float(1 - confidence_score)
        
        logger.info(f"Prediction for {file.filename}: {predicted_class} ({confidence:.2f})")
        
        return JSONResponse(content={
            "filename": file.filename,
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "raw_prediction": float(confidence_score),
            "image_size": f"{IMG_WIDTH}x{IMG_HEIGHT}"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
