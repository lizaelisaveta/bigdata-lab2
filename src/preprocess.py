import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cassandra_client import cassandra_client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_and_store_to_cassandra(data_path, img_height=150, img_width=150):
    logger.info("Connecting to Cassandra...")
    cassandra_client.connect()
    
    data_path = Path(data_path)
    image_files = list(data_path.glob('*.jpg'))
    
    if not image_files:
        logger.error(f"No images found in {data_path}")
        return 0, 0
    
    logger.info(f"Found {len(image_files)} images. Starting preprocessing...")
    
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            label = 1 if img_path.name.startswith('dog') else 0
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_width, img_height))
            img_array = np.array(img) / 255.0
            
            cassandra_client.insert_processed_data(
                img_path.name,
                label,
                img_array,
                img_height,
                img_width,
                3
            )
            processed += 1
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            errors += 1
    
    logger.info(f"Preprocessing completed! Processed: {processed}, Errors: {errors}")
    return processed, errors


def verify_processed_data():
    logger.info("Verifying processed data...")
    
    query = "SELECT COUNT(*) as count FROM processed_data"
    result = cassandra_client.session.execute(query).one()
    
    logger.info(f"Processed data count: {result.count}")
    return result.count


parser = argparse.ArgumentParser(description='Preprocess data and store to Cassandra')
parser.add_argument('--data_path', type=str, required=True, 
                    help='Path to directory with raw images')
parser.add_argument('--img_height', type=int, default=150,
                    help='Image height for preprocessing')
parser.add_argument('--img_width', type=int, default=150,
                    help='Image width for preprocessing')
parser.add_argument('--verify', action='store_true',
                    help='Verify processed data after processing')

args = parser.parse_args()

try:
    processed, errors = preprocess_and_store_to_cassandra(
        args.data_path, args.img_height, args.img_width
    )
    
    if args.verify:
        count = verify_processed_data()
        logger.info(f"Verification: {count} processed records found")
    
    logger.info(f"Summary: {processed} images processed, {errors} errors")
    
except Exception as e:
    logger.error(f"Script failed: {e}")
    raise
finally:
    cassandra_client.close()