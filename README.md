# Dogs vs Cats Classifier - MLOps Pipeline

Interaction with database for ML model classification (Dogs vs Cats) for BigData course

## 🚀 Tech Stack

- **ML Framework**: TensorFlow 2.x, Keras
- **API Framework**: FastAPI
- **Data Versioning**: DVC
- **Database**: Cassandra
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
│ ├── cassandra_client.py # Connection client for Cassandra
│ ├── config.py # Downloading configurations
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
• Get fork of repo.

• Pull data from DVC.

• Init Cassandra and update code for preprocessing and training.

• Created Dockerfile and docker-compose.yml, etc.

• Created piplines using GitHub Actions.

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/lizaelisaveta/bigdata-lab2.git
cd bigdata-lab2
```

### 2. Install Dependencies (MacOS)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.  Download Data
```bash
dvc pull
```

### 4. Run in Docker
```bash
docker-compose up --build
```

### 5. Run preprocessing in Docker
```bash
docker exec -it ml-api bash -c "python src/preprocess.py"
```

### 6. Run training of model in Docker
```bash
docker exec -it ml-api bash -c "python src/train.py"
```

### 7. Restart image in Docker
```bash
docker compose restart ml-api
```

## 🧪 Testing

Unit Tests
```bash
pytest tests/ -v
```

API Testing

Swagger UI: http://localhost:8000/docs

Model activity check: http://localhost:8000/active