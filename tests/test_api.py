import sys
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app

client = TestClient(app)


mock_model = MagicMock()
mock_model.predict.return_value = np.array([[0.8]]) 


@pytest.fixture(autouse=True)
def mock_model_load():
    with patch('src.api.load_model', return_value=mock_model):
        with patch('src.api.model', mock_model):
            yield


def test_activity_check():
    response = client.get("/active")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True


def test_predict_with_invalid_file():
    response = client.post("/predict/", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


def test_predict_with_valid_image():
    img = Image.new('RGB', (150, 150), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    files = {"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
    response = client.post("/predict/", files=files)
    
    assert response.status_code == 200
    json_response = response.json()
    assert "class" in json_response
    assert "confidence" in json_response
    assert json_response["class"] in ["Cat", "Dog"]


def test_predict_large_file():
    large_data = b"x" * (11 * 1024 * 1024)
    response = client.post("/predict/", files={"file": ("large.jpg", large_data, "image/jpeg")})
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]


def test_predict_without_model():
    with patch('src.api.model', None):
        img = Image.new('RGB', (150, 150), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        
        response = client.post("/predict/", files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")})
        assert response.status_code == 503
        assert "Model is not loaded" in response.json()["detail"]