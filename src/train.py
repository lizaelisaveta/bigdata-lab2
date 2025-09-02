import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import argparse
import os

from src.config import app_config


def create_model(input_shape=(150, 150, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


EPOCHS = int(app_config['model']['epochs'])
BATCH_SIZE = int(app_config['model']['batch_size'])
IMG_HEIGHT = int(app_config['model']['input_height'])
IMG_WIDTH = int(app_config['model']['input_width'])
MODEL_PATH = app_config['paths']['model_path']
DATA_PATH = app_config['paths']['processed_data']

parser = argparse.ArgumentParser(description='Train Dogs vs Cats CNN')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
args = parser.parse_args()

X = np.load(os.path.join(DATA_PATH, 'X.npy'))
y = np.load(os.path.join(DATA_PATH, 'y.npy'))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"Model training complete and saved to {MODEL_PATH}")