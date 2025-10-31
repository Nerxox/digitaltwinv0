"""MQTT Subscriber for ingesting IIoT data into TimescaleDB."""
import json
import logging
import signal
import sys
from typing import Dict, Any

import psycopg2
import paho.mqtt.client as mqtt
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ingestor")

# Database configuration
DB_CONFIG = {
    "dbname": "energydb",
    "user": "energy",
    "password": "energy",
    "host": "localhost",
    "port": "5432"
}

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
KEEPALIVE = 60
TOPICS = [f"factory/line1/{machine}/power" for machine in ["M1", "M2", "M3"]]


class DatabaseConnection:
    """Handles database connection and operations."""
    
    def __init__(self, config: Dict[str, str]):
        """Initialize database connection.
        
        Args:
            config: Database connection parameters
        """
        self.config = config
        self.conn = None
        self.cur = None
        self.connect()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.conn.autocommit = True
            self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def insert_reading(self, data: Dict[str, Any]) -> bool:
        """Insert a single reading into the database.
        
        Args:
            data: Dictionary containing machine reading data
            
        Returns:
            bool: True if insert was successful, False otherwise
        """
        query = """
        INSERT INTO energy_readings (machine_id, ts, power_kw, status)
        VALUES (%(machine_id)s, %(timestamp)s, %(power_kW)s, %(status)s)
        """
        
        try:
            self.cur.execute(query, data)
            return True
        except psycopg2.OperationalError as e:
            logger.error(f"Database error, attempting to reconnect: {e}")
            self.connect()  # Try to reconnect
            return False
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class MQTTSubscriber:
    """Handles MQTT subscription and message processing."""
    
    def __init__(self, broker: str, port: int, topics: list, db: DatabaseConnection):
        """Initialize MQTT subscriber.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            topics: List of topics to subscribe to
            db: Database connection instance
        """
        self.broker = broker
        self.port = port
        self.topics = topics
        self.db = db
        self.client = mqtt.Client()
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Set up MQTT client callbacks."""
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection event."""
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            # Subscribe to all specified topics
            for topic in self.topics:
                client.subscribe(topic, qos=0)
                logger.info(f"Subscribed to {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker with return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            # Parse message payload
            data = json.loads(msg.payload.decode("utf-8"))
            logger.debug(f"Received message on {msg.topic}: {data}")
            
            # Insert into database
            success = self.db.insert_reading(data)
            if not success:
                logger.warning(f"Failed to insert data into database: {data}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection. Will attempt to reconnect. RC: {rc}")
    
    def connect(self):
        """Connect to MQTT broker and start the loop."""
        try:
            self.client.connect(self.broker, self.port, 60)
            logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def start(self):
        """Start the MQTT client loop."""
        self.client.loop_forever()
    
    def stop(self):
        """Stop the MQTT client and clean up."""
        self.client.disconnect()
        self.db.close()
        logger.info("MQTT subscriber stopped")


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info("Shutdown signal received, cleaning up...")
    if 'subscriber' in globals():
        subscriber.stop()
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize database connection
        db = DatabaseConnection(DB_CONFIG)
        
        # Initialize and start MQTT subscriber
        subscriber = MQTTSubscriber(MQTT_BROKER, MQTT_PORT, TOPICS, db)
        subscriber.connect()
        
        logger.info("Starting MQTT subscriber...")
        subscriber.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'subscriber' in locals():
            subscriber.stop()
        logger.info("Application stopped")
