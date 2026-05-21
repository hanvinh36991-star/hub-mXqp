from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.schemas.base import BaseSchema, TimestampMixin, IDMixin


class KnowledgeBaseBase(BaseModel):
    """Base knowledge base schema"""
    name: str = Field(..., min_length=1, max_length=255, description="Knowledge base name")
    description: Optional[str] = Field(None, description="Knowledge base description")
    is_active: bool = Field(True, description="Whether the knowledge base is active")


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema for creating a knowledge base"""
    pass


class KnowledgeBaseUpdate(BaseModel):
    """Schema for updating a knowledge base"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class KnowledgeBaseInDB(IDMixin, TimestampMixin, KnowledgeBaseBase):
    """Knowledge base schema for database representation"""
    document_count: int = Field(0, description="Number of documents in the knowledge base")
    processed_document_count: int = Field(0, description="Number of processed documents")
    
    class Config:
        from_attributes = True


class KnowledgeBaseResponse(KnowledgeBaseInDB):
    """Knowledge base schema for API response"""
    pass


class KnowledgeBaseListResponse(BaseModel):
    """Response schema for listing knowledge bases"""
    items: List[KnowledgeBaseResponse]
    total: int
    page: int
    size: int
    pages: int