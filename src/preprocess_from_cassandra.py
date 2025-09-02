import numpy as np
from PIL import Image
import io
from tqdm import tqdm
from src.cassandra_client import cassandra_client


def preprocess_data_from_cassandra(img_height: int = 150, img_width: int = 150):    
    print("Connecting to Cassandra...")
    cassandra_client.connect()
    
    query = "SELECT * FROM raw_images"
    rows = cassandra_client.session.execute(query)
    
    print("Preprocessing images...")
    
    for row in tqdm(list(rows)):
        try:
            image = Image.open(io.BytesIO(row.image_data)).convert('RGB')
            
            image = image.resize((img_width, img_height))
            
            img_array = np.array(image) / 255.0
            
            label_int = 1 if row.label.lower() == 'dog' else 0
            
            cassandra_client.save_processed_data(
                image_id=str(row.image_id),
                filename=row.filename,
                label=label_int,
                normalized_data=img_array
            )
            
        except Exception as e:
            print(f"Error processing {row.filename}: {e}")
    
    print("Data preprocessing completed!")


preprocess_data_from_cassandra()