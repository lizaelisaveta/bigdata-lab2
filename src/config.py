import configparser
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.ini")

app_config = configparser.ConfigParser()
app_config.read(CONFIG_PATH)

IMG_HEIGHT = int(app_config["model"]["input_height"])
IMG_WIDTH = int(app_config["model"]["input_width"])
EPOCHS = int(app_config["model"]["epochs"])
BATCH_SIZE = int(app_config["model"]["batch_size"])
LEARNING_RATE = float(app_config["model"]["learning_rate"])

MODEL_PATH = app_config["paths"]["model_path"]
DATA_PATH = app_config["paths"]["data_path"]
RAW_TRAIN = app_config["paths"]["raw_train"]

API_HOST = app_config["api"]["host"]
API_PORT = int(app_config["api"]["port"])
MAX_FILE_SIZE = int(app_config["api"]["max_file_size"])
