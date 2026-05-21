from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from app.schemas.base import BaseSchema, TimestampMixin, IDMixin


class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"


class DocumentBase(BaseModel):
    """Base document schema"""
    knowledge_base_id: int = Field(..., description="ID of the knowledge base")
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    filepath: str = Field(..., description="Path to the stored file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")


class DocumentCreate(DocumentBase):
    """Schema for creating a document"""
    status: DocumentStatus = Field(DocumentStatus.UPLOADED, description="Processing status")


class DocumentUpdate(BaseModel):
    """Schema for updating a document"""
    status: Optional[DocumentStatus] = None
    processed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None


class DocumentInDB(IDMixin, TimestampMixin, DocumentBase):
    """Document schema for database representation"""
    status: DocumentStatus
    processed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentResponse(DocumentInDB):
    """Document schema for API response"""
    knowledge_base_name: Optional[str] = Field(None, description="Name of the knowledge base")
    chunk_count: int = Field(0, description="Number of chunks in the document")


class DocumentListResponse(BaseModel):
    """Response schema for listing documents"""
    items: List[DocumentResponse]
    total: int
    page: int
    size: int
    pages: int


class DocumentUploadRequest(BaseModel):
    """Request schema for uploading a document"""
    knowledge_base_id: int = Field(..., description="ID of the knowledge base")
    file_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional file metadata")


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload"""
    document_id: int
    filename: str
    filepath: str
    status: DocumentStatus
    message: str = "Document uploaded successfully"
    processing_queue_position: Optional[int] = None