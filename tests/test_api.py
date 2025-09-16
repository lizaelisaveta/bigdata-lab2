import pytest
from fastapi.testclient import TestClient
import sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api import app

client = TestClient(app)


def test_active():
    response = client.get("/active")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_predict_random_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(img_path)

    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

    assert response.status_code == 200