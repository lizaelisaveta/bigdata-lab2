import logging
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import os
import uuid
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cassandra_client import CassandraClient
from src.config import RAW_TRAIN, IMG_WIDTH, IMG_HEIGHT, DATA_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_and_store():
    client = CassandraClient()

    data_dir = Path(DATA_PATH) / "raw" / "train"
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} not found!")
        return

    image_paths = list(data_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} images in {data_dir}")

    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        label = 1 if img_path.name.startswith("dog") else 0
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_bytes = img_array.tobytes()

        img_id = uuid.uuid4()
        bucket = hash(img_id) % 10 

        client.insert_processed_data(bucket, img_id, label, img_bytes)

    logger.info("✅ Все изображения предобработаны и загружены в Cassandra")


preprocess_and_store()