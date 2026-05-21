import json
import time
import traceback
from typing import Dict, Any, List
from pathlib import Path
from kafka import KafkaConsumer
from sqlalchemy.orm import Session
from loguru import logger
from app.core.config import settings
from app.models.database import SessionLocal, DocumentStatus
from app.repositories.document import DocumentRepository
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.kafka_producer import KafkaProducerService
from app.services.models import ModelService


class DocumentWorker:
    """Worker for processing documents from Kafka queue"""
    
    def __init__(self):
        self.consumer = None
        self.document_processor = None
        self.vector_store = None
        self.kafka_producer = None
        self.model_service = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services"""
        try:
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                settings.kafka_process_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id="document_workers",
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                auto_offset_reset='earliest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                max_poll_records=1,  # Process one document at a time
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
            )
            
            # Initialize other services
            self.model_service = ModelService()
            self.document_processor = DocumentProcessor(self.model_service)
            self.vector_store = VectorStoreService()
            self.kafka_producer = KafkaProducerService()
            
            logger.info("Document worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document worker: {e}")
            raise
    
    def process_document(self, document_data: Dict[str, Any]) -> bool:
        """Process a single document"""
        document_id = document_data.get("document_id")
        filepath = document_data.get("filepath")
        filename = document_data.get("filename")
        knowledge_base_id = document_data.get("knowledge_base_id")
        
        logger.info(f"Starting processing for document {document_id}: {filename}")
        
        # Get database session
        db = SessionLocal()
        try:
            # Update document status to processing
            doc_repo = DocumentRepository(db)
            doc_repo.update_status(document_id, DocumentStatus.PROCESSING)
            
            # Check if file exists
            if not Path(filepath).exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Parse document with mineru
            output_dir = self.document_processor.parse_with_mineru(Path(filepath))
            
            # Find markdown files
            markdown_files = self.document_processor.find_markdown_files(output_dir)
            if not markdown_files:
                raise ValueError(f"No markdown files found in {output_dir}")
            
            # Process each markdown file
            all_chunks = []
            for markdown_file in markdown_files:
                # Split markdown into chunks
                chunks = self.document_processor.split_markdown_by_headers(markdown_file)
                
                # Process each chunk
                for idx, chunk in enumerate(chunks):
                    chunk_data = self.document_processor.process_chunk(
                        chunk=chunk,
                        document_id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        chunk_index=idx
                    )
                    all_chunks.append(chunk_data)
            
            # Insert chunks into vector store
            if all_chunks:
                vector_ids = self.vector_store.insert_chunks(all_chunks)
                logger.info(f"Inserted {len(vector_ids)} chunks into vector store")
            
            # Update document status to processed
            doc_repo.update_status(document_id, DocumentStatus.PROCESSED)
            
            # Send completion notification
            completion_message = {
                "document_id": document_id,
                "knowledge_base_id": knowledge_base_id,
                "filename": filename,
                "status": "processed",
                "chunk_count": len(all_chunks),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
            self.kafka_producer.send_processing_complete(completion_message)
            
            logger.info(f"Successfully processed document {document_id}: {filename}")
            return True
            
        except Exception as e:
            # Update document status to error
            if 'doc_repo' in locals() and document_id:
                doc_repo.update_status(
                    document_id, 
                    DocumentStatus.ERROR,
                    error_message=str(e)
                )
            
            # Send error notification
            error_message = {
                "document_id": document_id,
                "knowledge_base_id": knowledge_base_id,
                "filename": filename,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self.kafka_producer.send_error_notification(error_message)
            
            logger.error(f"Failed to process document {document_id}: {e}")
            logger.error(traceback.format_exc())
            return False
            
        finally:
            db.close()
    
    def run(self):
        """Run the worker continuously"""
        logger.info(f"Starting document worker, consuming from topic: {settings.kafka_process_topic}")
        
        try:
            for message in self.consumer:
                try:
                    self._start_time = time.time()
                    document_data = message.value
                    
                    logger.info(f"Received message: {document_data.get('document_id')}")
                    
                    # Process the document
                    success = self.process_document(document_data)
                    
                    if success:
                        logger.info(f"Successfully processed document {document_data.get('document_id')}")
                    else:
                        logger.error(f"Failed to process document {document_data.get('document_id')}")
                    
                    # Small sleep to prevent tight loop
                    time.sleep(0.1)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing message: {e}")
                    logger.error(traceback.format_exc())
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Worker failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the worker gracefully"""
        logger.info("Shutting down document worker...")
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        if self.kafka_producer:
            self.kafka_producer.close()
            logger.info("Kafka producer closed")
        
        logger.info("Document worker shutdown complete")
    
    def health_check(self) -> Dict[str, Any]:
        """Check worker health"""
        return {
            "status": "running" if self.consumer else "stopped",
            "topic": settings.kafka_process_topic,
            "services": {
                "kafka_consumer": self.consumer is not None,
                "vector_store": self.vector_store is not None,
                "document_processor": self.document_processor is not None,
                "kafka_producer": self.kafka_producer is not None
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


def main():
    """Main entry point for document worker"""
    worker = None
    try:
        worker = DocumentWorker()
        worker.run()
    except Exception as e:
        logger.error(f"Failed to start document worker: {e}")
        logger.error(traceback.format_exc())
        if worker:
            worker.shutdown()
        raise


if __name__ == "__main__":
    main()