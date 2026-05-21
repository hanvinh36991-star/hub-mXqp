import json
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from app.core.config import settings


class KafkaProducerService:
    """Service for producing Kafka messages"""
    
    def __init__(self):
        self.producer = None
        self._connect()
    
    def _connect(self):
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',  # Wait for all in-sync replicas to acknowledge
                retries=3,   # Retry up to 3 times
                max_in_flight_requests_per_connection=1,  # Maintain order
                request_timeout_ms=30000,  # 30 second timeout
            )
            logger.info(f"Connected to Kafka at {settings.kafka_bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.producer = None
    
    def send_document_for_processing(self, message: Dict[str, Any]) -> bool:
        """Send document to Kafka for processing"""
        return self._send_message(
            topic=settings.kafka_process_topic,
            key=str(message.get("document_id", "unknown")),
            value=message,
            message_type="document_processing"
        )
    
    def send_error_notification(self, error_data: Dict[str, Any]) -> bool:
        """Send error notification to Kafka"""
        return self._send_message(
            topic=settings.kafka_upload_topic,
            key=error_data.get("document_id", "error"),
            value=error_data,
            message_type="error_notification"
        )
    
    def send_processing_complete(self, completion_data: Dict[str, Any]) -> bool:
        """Send processing completion notification"""
        return self._send_message(
            topic=settings.kafka_upload_topic,
            key=str(completion_data.get("document_id", "complete")),
            value=completion_data,
            message_type="processing_complete"
        )
    
    def _send_message(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any],
        message_type: str
    ) -> bool:
        """Send a message to Kafka topic"""
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            # Add metadata to message
            enhanced_message = {
                **value,
                "_metadata": {
                    "message_type": message_type,
                    "timestamp": value.get("timestamp"),
                    "version": "1.0"
                }
            }
            
            # Send message
            future = self.producer.send(
                topic=topic,
                key=key.encode('utf-8'),
                value=enhanced_message
            )
            
            # Wait for send to complete (optional, can be async)
            result = future.get(timeout=10)
            
            logger.info(
                f"Sent {message_type} message to topic {topic}, "
                f"partition {result.partition}, offset {result.offset}"
            )
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending {message_type} message: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send {message_type} message: {e}")
            return False
    
    def close(self):
        """Close Kafka producer connection"""
        if self.producer:
            self.producer.flush(timeout=10)
            self.producer.close()
            logger.info("Kafka producer closed")
    
    def is_connected(self) -> bool:
        """Check if producer is connected"""
        return self.producer is not None
    
    def get_topic_metadata(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for Kafka topics"""
        if not self.producer:
            return {"error": "Producer not connected"}
        
        try:
            if topic:
                topics = [topic]
            else:
                topics = [settings.kafka_process_topic, settings.kafka_upload_topic]
            
            metadata = {}
            for topic_name in topics:
                partitions = self.producer.partitions_for(topic_name)
                if partitions:
                    metadata[topic_name] = {
                        "partition_count": len(partitions),
                        "partitions": list(partitions)
                    }
                else:
                    metadata[topic_name] = {"error": "Topic not found"}
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get topic metadata: {e}")
            return {"error": str(e)}