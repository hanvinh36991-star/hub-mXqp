from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "Multimodal RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    database_url: str = "sqlite:///./multimodal_rag.db"
    
    # Milvus Vector Database
    milvus_uri: str = ""
    milvus_token: str = ""
    milvus_collection: str = "multimodal_rag_data"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_upload_topic: str = "rag_document_upload"
    kafka_process_topic: str = "rag_document_process"
    
    # File Storage
    upload_dir: str = "./uploads"
    processed_dir: str = "./processed"
    
    # Model APIs
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-flash"
    
    # Local Model Paths
    bge_model_path: str = ""
    clip_model_path: str = ""
    
    # Mineru Document Parser
    mineru_endpoint: str = "http://localhost:30000"
    
    # Retrieval Settings
    retrieval_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("upload_dir", "processed_dir", pre=True)
    def create_dirs(cls, v):
        """Create directories if they don't exist"""
        os.makedirs(v, exist_ok=True)
        return v
    
    @property
    def database_engine_args(self):
        """Get database engine arguments"""
        return {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }


settings = Settings()