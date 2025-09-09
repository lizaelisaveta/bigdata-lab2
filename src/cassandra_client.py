import os
import logging
import uuid
import numpy as np
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from cassandra.policies import DCAwareRoundRobinPolicy


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CassandraClient:
    def __init__(self):
        host = os.getenv("CASSANDRA_HOST", "localhost")
        port = int(os.getenv("CASSANDRA_PORT", "9042"))
        username = os.getenv("CASSANDRA_USERNAME", "cassandra")
        password = os.getenv("CASSANDRA_PASSWORD", "cassandra")
        keyspace = os.getenv("CASSANDRA_KEYSPACE", "ml_results")

        auth_provider = PlainTextAuthProvider(username=username, password=password)

        contact_points = [h.strip() for h in host.split(",") if h.strip()]

        self.cluster = Cluster(
            contact_points,
            port=port,
            auth_provider=auth_provider,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
            protocol_version=5 
        )

        self.session = self.cluster.connect()
        self.keyspace = keyspace
        self.create_keyspace()
        self.session.set_keyspace(self.keyspace)
        self.create_tables()

        logger.info(f"✅ Connected to Cassandra at {contact_points}:{port}, keyspace: {self.keyspace}")

    def create_keyspace(self):
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{ 'class': 'SimpleStrategy', 'replication_factor': '1' }}
        """)

    def create_tables(self):
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS processed_data (
                bucket int,
                id uuid,
                label int,
                image blob,
                PRIMARY KEY ((bucket), id)
            )
        """)

        self.session.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id uuid PRIMARY KEY,
                label text,
                confidence float
            )
        """)

    def insert_processed_data(self, bucket, id, label, image):
        self.session.execute(
            "INSERT INTO processed_data (bucket, id, label, image) VALUES (%s, %s, %s, %s)",
            (bucket, id, label, image)
        )

    def fetch_processed_data(self, bucket, batch_size=500):
        stmt = SimpleStatement(
            "SELECT label, image FROM processed_data WHERE bucket = %s",
            fetch_size=batch_size
        )
        return self.session.execute(stmt, (bucket,))
    
    def insert_prediction(self, prediction_id, label, confidence):
        query = """
        INSERT INTO predictions (id, label, confidence)
        VALUES (%s, %s, %s)
        """
        self.session.execute(query, (prediction_id, label, confidence))
