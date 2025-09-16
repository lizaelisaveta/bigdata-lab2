import logging
import numpy as np
from tensorflow import keras
from keras import layers, models
from cassandra.query import SimpleStatement
import sys, os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cassandra_client import CassandraClient
from src.config import IMG_WIDTH, IMG_HEIGHT, EPOCHS, BATCH_SIZE, MODEL_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stream_batches_from_cassandra(batch_size=500, max_buckets=50):
    client = CassandraClient()

    total_images = 0
    used_buckets = 0

    for bucket in tqdm(range(max_buckets), desc="📥 Загрузка данных из Cassandra"):
        stmt = SimpleStatement(
            "SELECT label, image FROM processed_data WHERE bucket = %s",
            fetch_size=batch_size
        )
        rows = list(client.session.execute(stmt, (bucket,)))

        if not rows:
            continue

        used_buckets += 1
        total_images += len(rows)

        for row in rows:
            try:
                img_array = np.frombuffer(row.image, dtype=np.float32)
                img_array = img_array.reshape((IMG_HEIGHT, IMG_WIDTH, 3))
                X = np.expand_dims(img_array, axis=0)
                y = np.array([row.label], dtype=np.int32)
                yield X, y
            except Exception as e:
                logger.error(f"❌ Ошибка при обработке строки: {e}")

    logger.info(f"📊 Итог: использовано {used_buckets} бакетов, загружено {total_images} изображений")


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model():
    model = build_model()
    logger.info("🚀 Начало обучения модели")

    for epoch in range(EPOCHS): 
        logger.info(f"===== Эпоха {epoch+1}/{EPOCHS} =====")

        total_loss, total_acc, batches = 0.0, 0.0, 0
        for X, y in stream_batches_from_cassandra(batch_size=BATCH_SIZE):
            metrics = model.train_on_batch(X, y, return_dict=True)
            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
            batches += 1

        epoch_loss = total_loss / batches if batches > 0 else 0
        epoch_acc = total_acc / batches if batches > 0 else 0

        logger.info(f"🎯 Epoch {epoch+1}: accuracy={epoch_acc:.4f}, loss={epoch_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"✅ Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    train_model()