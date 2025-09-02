# Dogs vs Cats Classifier - MLOps Pipeline

CI/CD pipeline for ML model classification (Dogs vs Cats) for BigData course

## 🚀 Tech Stack

- **ML Framework**: TensorFlow 2.x, Keras
- **API Framework**: FastAPI
- **Data Versioning**: DVC
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Testing**: pytest, FastAPI TestClient

## 📊 Dataset

Dataset: Dogs vs Cats from Kaggle
- **Source**: https://www.kaggle.com/c/dogs-vs-cats
- **Size**: 25,000 images (12,500 dogs, 12,500 cats)
- **Task**: Binary image classification

## ⚙️ Project Structure
```
bigdata-lab/
├── src/
│ ├── api.py # FastAPI application
│ ├── train.py # Model training script
│ └── preprocess.py # Data preprocessing
├── tests/
│ └── test_api.py # API tests
├── notebooks/
│ └── train.ipynb # Model training notebook
├── models/ # Trained models (DVC)
├── data/ # Raw and processed data (DVC)
├── .github/workflows/ # CI/CD pipelines
└── config files # Docker, requirements...
```

## Workflow
• Downloaded dataset from Kaggle.

• Preprocess data and train model.

• Transformed research notebook into scripts.

• Put dataset and model using DVC.

• Created Dockerfile and docker-compose.yml, etc.

• Created piplines using GitHub Actions.

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/lizaelisaveta/bigdata-lab.git
cd bigdata-lab
```

### 2. Install Dependencies (MacOS)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.  Download Data & Model
```bash
dvc pull
```

### 4. Run API Locally
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run in Docker
```bash
docker-compose up --build
```


## 🧪 Testing

Unit Tests
```bash
pytest tests/ -v
```

API Testing

Swagger UI: http://localhost:8000/docs

Model activity check: http://localhost:8000/active