import os
import logging
import uuid
import numpy as np
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from cassandra.policies import DCAwareRoundRobinPolicy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CassandraClient:
    def __init__(self):
        self.host = os.getenv('CASSANDRA_HOST', 'localhost')
        self.port = int(os.getenv('CASSANDRA_PORT', 9042))
        self.username = os.getenv('CASSANDRA_USERNAME', 'cassandra')
        self.password = os.getenv('CASSANDRA_PASSWORD', 'cassandra')
        self.keyspace = os.getenv('CASSANDRA_KEYSPACE', 'ml_results')
        self.session = None
        self.cluster = None

    def connect(self):
        try:
            auth_provider = PlainTextAuthProvider(
                username=self.username, 
                password=self.password
            )
            
            self.cluster = Cluster(
                [self.host], 
                port=self.port, 
                auth_provider=auth_provider,
                load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'),
                protocol_version=4
            )
            self.session = self.cluster.connect()

            self.create_keyspace()
            self.session.set_keyspace(self.keyspace)
            self.create_tables()
            
            logger.info(f"Connected to Cassandra at {self.host}:{self.port}, keyspace: {self.keyspace}")

        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            raise

    def create_keyspace(self):
        create_ks_query = f"""
        CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
        WITH replication = {{
            'class': 'SimpleStrategy',
            'replication_factor': '1'
        }}
        """
        self.session.execute(create_ks_query)

    def create_tables(self):
        self.session.set_keyspace(self.keyspace)

        create_processed_table = """
        CREATE TABLE IF NOT EXISTS processed_data (
            image_id UUID PRIMARY KEY,
            filename TEXT,
            label INT,
            pixel_data LIST<FLOAT>,
            height INT,
            width INT,
            channels INT,
            created_at TIMESTAMP
        )
        """
        self.session.execute(create_processed_table)
        logger.info("Table 'processed_data' is ready.")

        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id UUID PRIMARY KEY,
            filename TEXT,
            predicted_class TEXT,
            confidence FLOAT,
            raw_prediction FLOAT,
            model_version TEXT,
            prediction_timestamp TIMESTAMP
        )
        """
        self.session.execute(create_predictions_table)
        logger.info("Table 'predictions' is ready.")

    def insert_processed_data(self, filename, label, pixel_array, height, width, channels):
        if not self.session:
            raise Exception("Session is not initialized. Call connect() first.")

        pixel_list = pixel_array.flatten().tolist()
        
        query = """
        INSERT INTO processed_data (image_id, filename, label, pixel_data, height, width, channels, created_at)
        VALUES (uuid(), %s, %s, %s, %s, %s, %s, toTimestamp(now()))
        """
        try:
            self.session.execute(query, (filename, label, pixel_list, height, width, channels))
            logger.debug(f"Processed data for {filename} saved to Cassandra.")
        except Exception as e:
            logger.error(f"Failed to insert processed data for {filename}: {e}")
            raise

    def get_all_processed_data(self):
        if not self.session:
            raise Exception("Session is not initialized. Call connect() first.")

        try:
            query = "SELECT image_id, label, pixel_data, height, width, channels FROM processed_data"
            rows = self.session.execute(query)
            
            images = []
            labels = []
            
            for row in rows:
                pixel_array = np.array(row.pixel_data, dtype=np.float32)
                image_shape = (row.height, row.width, row.channels)
                image = pixel_array.reshape(image_shape)
                
                images.append(image)
                labels.append(row.label)
            
            X = np.array(images)
            y = np.array(labels)
            
            logger.info(f"Loaded {len(X)} processed samples from Cassandra")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to get processed data: {e}")
            raise

    def insert_prediction(self, filename, predicted_class, confidence, raw_prediction, model_version="v1.0.0"):
        if not self.session:
            raise Exception("Session is not initialized. Call connect() first.")

        query = """
        INSERT INTO predictions (prediction_id, filename, predicted_class, confidence, raw_prediction, model_version, prediction_timestamp)
        VALUES (uuid(), %s, %s, %s, %s, %s, toTimestamp(now()))
        """
        try:
            self.session.execute(query, (
                filename,
                predicted_class,
                confidence,
                raw_prediction,
                model_version
            ))
            logger.info(f"Prediction for {filename} saved to Cassandra.")
        except Exception as e:
            logger.error(f"Failed to insert prediction: {e}")
            raise

    def close(self):
        if self.cluster:
            self.cluster.shutdown()
            logger.info("Cassandra connection closed.")


cassandra_client = CassandraClient()