import configparser
import os
from pathlib import Path


def load_config():
    config = configparser.ConfigParser()
    
    config_path = Path(__file__).parent.parent / 'config.ini'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config.read(config_path)
    return config


app_config = load_config()