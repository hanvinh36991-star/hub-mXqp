from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pymilvus import MilvusClient, DataType
from loguru import logger
from app.core.config import settings


class VectorStoreService:
    """Service for vector storage and retrieval using Milvus"""
    
    def __init__(self):
        self.client = None
        self.collection_name = settings.milvus_collection
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            self.client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token
            )
            logger.info(f"Connected to Milvus at {settings.milvus_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper schema"""
        if not self.client.has_collection(self.collection_name):
            self._create_collection()
        else:
            logger.info(f"Collection {self.collection_name} already exists")
    
    def _create_collection(self):
        """Create collection with schema"""
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="document_id", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.INT64)
        
        # Vector fields
        schema.add_field(field_name="bge_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
        schema.add_field(field_name="clip_text_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="clip_image_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        
        # Text fields
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        
        # Create collection
        index_params = MilvusClient.prepare_index_params()
        
        # Index for BGE vectors (for text search)
        index_params.add_index(
            field_name="bge_vector",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 1024}
        )
        
        # Index for CLIP vectors (for multimodal search)
        index_params.add_index(
            field_name="clip_text_vector",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 1024}
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Created collection {self.collection_name}")
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Insert document chunks into vector store"""
        try:
            # Prepare data for insertion
            data = []
            for chunk in chunks:
                row = {
                    "document_id": chunk.get("document_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "knowledge_base_id": chunk.get("knowledge_base_id"),
                    "bge_vector": chunk.get("bge_vector", []),
                    "clip_text_vector": chunk.get("clip_text_vector", []),
                    "clip_image_vector": chunk.get("clip_image_vector", []),
                    "content": chunk.get("content", ""),
                    "content_type": chunk.get("content_type", "text"),
                    "metadata": chunk.get("metadata", {})
                }
                data.append(row)
            
            # Insert data
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            logger.info(f"Inserted {len(result['ids'])} chunks into vector store")
            return result['ids']
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        vector_field: str = "bge_vector",
        knowledge_base_id: Optional[int] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": "IP",  # Inner Product (cosine similarity)
                "params": {"nprobe": 10}
            }
            
            # Prepare filter
            expr = None
            if knowledge_base_id:
                expr = f"knowledge_base_id == {knowledge_base_id}"
            
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        values_str = ", ".join([str(v) for v in value])
                        filter_parts.append(f"{key} in [{values_str}]")
                    else:
                        filter_parts.append(f"{key} == {value}")
                
                filter_expr = " and ".join(filter_parts)
                expr = f"({expr}) and ({filter_expr})" if expr else filter_expr
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field=vector_field,
                search_params=search_params,
                limit=top_k,
                output_fields=["id", "document_id", "chunk_id", "content", "content_type", "metadata"],
                expr=expr
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit["id"],
                        "document_id": hit["entity"]["document_id"],
                        "chunk_id": hit["entity"]["chunk_id"],
                        "content": hit["entity"]["content"],
                        "content_type": hit["entity"]["content_type"],
                        "metadata": hit["entity"].get("metadata", {}),
                        "score": hit["distance"]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete_by_document(self, document_id: int) -> int:
        """Delete all chunks for a document"""
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f"document_id == {document_id}"
            )
            
            deleted_count = result.get("delete_count", 0)
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            raise
    
    def delete_by_knowledge_base(self, knowledge_base_id: int) -> int:
        """Delete all chunks for a knowledge base"""
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f"knowledge_base_id == {knowledge_base_id}"
            )
            
            deleted_count = result.get("delete_count", 0)
            logger.info(f"Deleted {deleted_count} chunks for knowledge base {knowledge_base_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for knowledge base {knowledge_base_id}: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}