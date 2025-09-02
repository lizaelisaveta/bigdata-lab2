import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import logging


logger = logging.getLogger(__name__)


class CassandraClient:
    def __init__(self):
        self.host = os.getenv('CASSANDRA_HOST', 'localhost')
        self.port = int(os.getenv('CASSANDRA_PORT', 9042))
        self.keyspace = os.getenv('CASSANDRA_KEYSPACE', 'ml_results')
        self.session = None
        
    def connect(self):
        try:
            auth_provider = PlainTextAuthProvider(
                username=os.getenv('CASSANDRA_USERNAME'),
                password=os.getenv('CASSANDRA_PASSWORD')
            ) if os.getenv('CASSANDRA_USERNAME') else None
            
            cluster = Cluster(
                [self.host],
                port=self.port,
                auth_provider=auth_provider
            )
            
            self.session = cluster.connect()
            self._create_keyspace()
            self._create_tables()
            logger.info("Connected to Cassandra successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            raise

    def _create_keyspace(self):
        create_keyspace = """
        CREATE KEYSPACE IF NOT EXISTS ml_results 
        WITH replication = {
            'class': 'SimpleStrategy', 
            'replication_factor': 1
        }
        """
        self.session.execute(create_keyspace)
        self.session.set_keyspace(self.keyspace)

    def _create_tables(self):
        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id UUID PRIMARY KEY,
            filename TEXT,
            predicted_class TEXT,
            confidence FLOAT,
            raw_prediction FLOAT,
            image_size TEXT,
            created_at TIMESTAMP
        )
        """
        self.session.execute(create_predictions_table)


    def save_prediction(self, prediction_data):
        try:
            insert_query = """
            INSERT INTO predictions (
                prediction_id, filename, predicted_class, 
                confidence, raw_prediction, image_size, created_at
            ) VALUES (uuid(), %s, %s, %s, %s, %s, toTimestamp(now()))
            """
            
            self.session.execute(insert_query, (
                prediction_data['filename'],
                prediction_data['class'],
                prediction_data['confidence'],
                prediction_data['raw_prediction'],
                prediction_data['image_size']
            ))
            
            logger.info(f"Prediction saved to Cassandra: {prediction_data['filename']}")
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            raise


cassandra_client = CassandraClient()