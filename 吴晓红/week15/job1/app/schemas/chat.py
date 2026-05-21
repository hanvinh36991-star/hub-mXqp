from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from app.schemas.base import BaseSchema, TimestampMixin, IDMixin


class MessageRole(str, Enum):
    """Message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatRequest(BaseModel):
    """Request schema for chat"""
    knowledge_base_id: int = Field(..., description="ID of the knowledge base to query")
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID (creates new if not provided)")
    stream: bool = Field(False, description="Whether to stream the response")
    
    # Model parameters
    model: Optional[str] = Field(None, description="Model to use for generation")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2000, ge=1, le=8000, description="Maximum tokens to generate")
    
    # Retrieval parameters
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Similarity threshold for retrieval")


class ChatResponse(BaseModel):
    """Response schema for chat"""
    session_id: str = Field(..., description="Chat session ID")
    message_id: int = Field(..., description="Message ID")
    response: str = Field(..., description="Assistant response")
    
    # Metadata
    model_used: str = Field(..., description="Model used for generation")
    tokens_used: int = Field(0, description="Tokens used in the response")
    processing_time: float = Field(..., description="Time taken to process (seconds)")
    
    # Retrieval information
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved chunks")
    retrieval_scores: List[float] = Field(default_factory=list, description="Similarity scores")
    source_documents: List[str] = Field(default_factory=list, description="Source document names")


class ChatSessionBase(BaseModel):
    """Base chat session schema"""
    knowledge_base_id: Optional[int] = Field(None, description="ID of the knowledge base")
    user_id: Optional[str] = Field(None, description="User ID")
    title: Optional[str] = Field(None, description="Session title")


class ChatSessionCreate(ChatSessionBase):
    """Schema for creating a chat session"""
    pass


class ChatSessionInDB(IDMixin, TimestampMixin, ChatSessionBase):
    """Chat session schema for database representation"""
    session_id: str = Field(..., description="Unique session ID")
    
    class Config:
        from_attributes = True


class ChatSessionResponse(ChatSessionInDB):
    """Chat session schema for API response"""
    message_count: int = Field(0, description="Number of messages in the session")


class ChatMessageBase(BaseModel):
    """Base chat message schema"""
    session_id: int = Field(..., description="Session ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatMessageCreate(ChatMessageBase):
    """Schema for creating a chat message"""
    retrieved_chunk_ids: Optional[str] = Field(None, description="JSON array of retrieved chunk IDs")
    model_used: Optional[str] = Field(None, description="Model used for generation")
    tokens_used: Optional[int] = Field(None, description="Tokens used")


class ChatMessageInDB(IDMixin, ChatMessageBase):
    """Chat message schema for database representation"""
    retrieved_chunk_ids: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        from_attributes = True


class ChatMessageResponse(ChatMessageInDB):
    """Chat message schema for API response"""
    pass


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history"""
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]