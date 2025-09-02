import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from src.cassandra_client import cassandra_client


def load_dataset_to_cassandra(raw_data_path: str):    
    print("Connecting to Cassandra...")
    cassandra_client.connect()
    
    raw_data_path = Path(raw_data_path)
    image_files = list(raw_data_path.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images. Loading to Cassandra...")
    
    for img_file in tqdm(image_files):
        try:
            label = 'dog' if img_file.name.startswith('dog') else 'cat'
            
            with open(img_file, 'rb') as f:
                image_data = f.read()
            
            img = Image.open(img_file)
            width, height = img.size
            
            cassandra_client.save_raw_image(
                filename=img_file.name,
                label=label,
                image_data=image_data,
                width=width,
                height=height
            )
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print("Dataset loaded to Cassandra successfully!")


load_dataset_to_cassandra('data/raw/train')