from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from app.models.database import KnowledgeBase, Document
from app.repositories.base import BaseRepository


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase, Dict, Dict]):
    """Repository for knowledge base operations"""
    
    def __init__(self, db: Session):
        super().__init__(KnowledgeBase, db)
    
    def get_active_bases(self, skip: int = 0, limit: int = 100) -> List[KnowledgeBase]:
        """Get active knowledge bases"""
        return (
            self.db.query(KnowledgeBase)
            .filter(KnowledgeBase.is_active == True)
            .order_by(desc(KnowledgeBase.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_name(self, name: str) -> Optional[KnowledgeBase]:
        """Get knowledge base by name"""
        return self.db.query(KnowledgeBase).filter(KnowledgeBase.name == name).first()
    
    def get_with_documents(self, knowledge_base_id: int) -> Optional[KnowledgeBase]:
        """Get knowledge base with its documents"""
        return (
            self.db.query(KnowledgeBase)
            .filter(KnowledgeBase.id == knowledge_base_id)
            .first()
        )
    
    def search(self, query: str, skip: int = 0, limit: int = 100) -> List[KnowledgeBase]:
        """Search knowledge bases by name or description"""
        search_term = f"%{query}%"
        return (
            self.db.query(KnowledgeBase)
            .filter(
                or_(
                    KnowledgeBase.name.ilike(search_term),
                    KnowledgeBase.description.ilike(search_term)
                )
            )
            .filter(KnowledgeBase.is_active == True)
            .order_by(desc(KnowledgeBase.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_document_count(self, knowledge_base_id: int) -> int:
        """Get number of documents in a knowledge base"""
        return (
            self.db.query(Document)
            .filter(Document.knowledge_base_id == knowledge_base_id)
            .count()
        )
    
    def get_processed_document_count(self, knowledge_base_id: int) -> int:
        """Get number of processed documents in a knowledge base"""
        return (
            self.db.query(Document)
            .filter(
                Document.knowledge_base_id == knowledge_base_id,
                Document.status == "processed"
            )
            .count()
        )