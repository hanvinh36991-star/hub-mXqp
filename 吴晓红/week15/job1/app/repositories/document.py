from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from app.models.database import Document, DocumentChunk
from app.repositories.base import BaseRepository
from app.schemas.document import DocumentStatus


class DocumentRepository(BaseRepository[Document, Dict, Dict]):
    """Repository for document operations"""
    
    def __init__(self, db: Session):
        super().__init__(Document, db)
    
    def get_by_knowledge_base(
        self, 
        knowledge_base_id: int, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[DocumentStatus] = None
    ) -> List[Document]:
        """Get documents in a knowledge base"""
        query = (
            self.db.query(Document)
            .filter(Document.knowledge_base_id == knowledge_base_id)
        )
        
        if status:
            query = query.filter(Document.status == status)
        
        return (
            query.order_by(desc(Document.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_status(self, status: DocumentStatus, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get documents by status"""
        return (
            self.db.query(Document)
            .filter(Document.status == status)
            .order_by(Document.created_at)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_processing_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get documents that are being processed"""
        return (
            self.db.query(Document)
            .filter(Document.status == DocumentStatus.PROCESSING)
            .order_by(Document.created_at)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_processed_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get processed documents"""
        return (
            self.db.query(Document)
            .filter(Document.status == DocumentStatus.PROCESSED)
            .order_by(desc(Document.processed_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_with_chunks(self, document_id: int) -> Optional[Document]:
        """Get document with its chunks"""
        return (
            self.db.query(Document)
            .filter(Document.id == document_id)
            .first()
        )
    
    def search(
        self, 
        query: str, 
        knowledge_base_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """Search documents by filename or title"""
        search_term = f"%{query}%"
        
        query_builder = self.db.query(Document).filter(
            or_(
                Document.filename.ilike(search_term),
                Document.title.ilike(search_term)
            )
        )
        
        if knowledge_base_id:
            query_builder = query_builder.filter(Document.knowledge_base_id == knowledge_base_id)
        
        return (
            query_builder.order_by(desc(Document.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_status(
        self, 
        document_id: int, 
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        """Update document status"""
        document = self.get(document_id)
        if document:
            document.status = status
            if status == DocumentStatus.PROCESSED:
                from datetime import datetime
                document.processed_at = datetime.now()
            elif status == DocumentStatus.ERROR and error_message:
                document.processing_error = error_message
            
            self.db.commit()
            self.db.refresh(document)
        
        return document
    
    def get_document_stats(self, knowledge_base_id: Optional[int] = None) -> Dict[str, int]:
        """Get document statistics"""
        query = self.db.query(Document.status, Document.knowledge_base_id)
        
        if knowledge_base_id:
            query = query.filter(Document.knowledge_base_id == knowledge_base_id)
        
        results = query.all()
        
        stats = {
            "total": 0,
            "uploaded": 0,
            "processing": 0,
            "processed": 0,
            "error": 0
        }
        
        for status, kb_id in results:
            if knowledge_base_id is None or kb_id == knowledge_base_id:
                stats["total"] += 1
                if status == DocumentStatus.UPLOADED:
                    stats["uploaded"] += 1
                elif status == DocumentStatus.PROCESSING:
                    stats["processing"] += 1
                elif status == DocumentStatus.PROCESSED:
                    stats["processed"] += 1
                elif status == DocumentStatus.ERROR:
                    stats["error"] += 1
        
        return stats


class DocumentChunkRepository(BaseRepository[DocumentChunk, Dict, Dict]):
    """Repository for document chunk operations"""
    
    def __init__(self, db: Session):
        super().__init__(DocumentChunk, db)
    
    def get_by_document(self, document_id: int, skip: int = 0, limit: int = 1000) -> List[DocumentChunk]:
        """Get chunks for a document"""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_content_type(self, document_id: int, content_type: str) -> List[DocumentChunk]:
        """Get chunks by content type"""
        return (
            self.db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.content_type == content_type
            )
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
    
    def get_chunks_with_images(self, document_id: int) -> List[DocumentChunk]:
        """Get chunks that contain images"""
        return (
            self.db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.has_images == True
            )
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
    
    def count_by_document(self, document_id: int) -> int:
        """Count chunks in a document"""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .count()
        )
    
    def delete_by_document(self, document_id: int) -> int:
        """Delete all chunks for a document"""
        count = (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .delete()
        )
        self.db.commit()
        return count