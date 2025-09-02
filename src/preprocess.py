import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm


def load_and_preprocess_data(raw_data_path, img_height=150, img_width=150):
    images = []
    labels = []

    raw_data_path = Path(raw_data_path)
    for img_file in tqdm(raw_data_path.glob('*.jpg')):
        label = 1 if img_file.name.startswith('dog') else 0

        img = Image.open(img_file).convert('RGB')
        img = img.resize((img_width, img_height))

        img_array = np.array(img) / 255.0

        images.append(img_array)
        labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    return X, y


RAW_DATA_PATH = 'data/raw/train'
PROCESSED_DATA_PATH = 'data/processed'
IMG_HEIGHT = 150
IMG_WIDTH = 150

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

print("Loading and preprocessing data...")
X, y = load_and_preprocess_data(RAW_DATA_PATH, IMG_HEIGHT, IMG_WIDTH)

np.save(os.path.join(PROCESSED_DATA_PATH, 'X.npy'), X)
np.save(os.path.join(PROCESSED_DATA_PATH, 'y.npy'), y)
print(f"Data saved to {PROCESSED_DATA_PATH}")
print(f"X shape: {X.shape}, y shape: {y.shape}")