from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from app.core.config import settings

Base = declarative_base()


class KnowledgeBase(Base):
    """Knowledge base table - represents a collection of documents"""
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")


class Document(Base):
    """Document table - represents uploaded documents"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    
    # Processing status
    status = Column(String(50), default="uploaded")  # uploaded, processing, processed, error
    processed_at = Column(DateTime)
    processing_error = Column(Text)
    
    # Metadata
    page_count = Column(Integer)
    language = Column(String(50))
    title = Column(String(500))
    author = Column(String(255))
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Document chunk table - represents processed chunks of documents"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    content_type = Column(String(50), nullable=False)  # text, image, mixed
    content = Column(Text, nullable=False)
    
    # Vector IDs (stored in Milvus)
    bge_vector_id = Column(String(100))
    clip_text_vector_id = Column(String(100))
    clip_image_vector_id = Column(String(100))
    
    # Metadata
    page_number = Column(Integer)
    section_title = Column(String(500))
    has_images = Column(Boolean, default=False)
    image_paths = Column(Text)  # JSON array of image paths
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    """Chat session table - stores chat history"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"))
    user_id = Column(String(100))
    
    title = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat message table - stores individual messages"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Metadata for RAG responses
    retrieved_chunk_ids = Column(Text)  # JSON array of chunk IDs
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


# Create database engine and session
engine = create_engine(settings.database_url, **settings.database_engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()