from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from app.models.database import get_db
from app.repositories.document import DocumentRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentStatus,
    DocumentUploadRequest,
    DocumentUploadResponse
)
from app.core.config import settings
from app.services.kafka_producer import KafkaProducerService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png'}


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: Optional[int] = Query(None, description="Filter by knowledge base ID"),
    status: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    search: Optional[str] = Query(None, description="Search query for filename or title"),
    db: Session = Depends(get_db)
):
    """List documents"""
    repo = DocumentRepository(db)
    kb_repo = KnowledgeBaseRepository(db)
    
    # Build filters
    filters = {}
    if knowledge_base_id:
        filters["knowledge_base_id"] = knowledge_base_id
    if status:
        filters["status"] = status
    
    if search:
        items = repo.search(query=search, knowledge_base_id=knowledge_base_id, skip=skip, limit=limit)
        total = repo.count(filters)
    else:
        items = repo.get_multi(skip=skip, limit=limit, filters=filters)
        total = repo.count(filters)
    
    # Enhance items with additional info
    enhanced_items = []
    for item in items:
        item_dict = {**item.__dict__}
        item_dict.pop('_sa_instance_state', None)
        
        # Get knowledge base name
        kb_name = None
        if item.knowledge_base_id:
            kb = kb_repo.get(item.knowledge_base_id)
            if kb:
                kb_name = kb.name
        
        # Get chunk count (would need DocumentChunkRepository)
        chunk_count = 0
        
        enhanced_items.append(DocumentResponse(
            **item_dict,
            knowledge_base_name=kb_name,
            chunk_count=chunk_count
        ))
    
    pages = (total + limit - 1) // limit if limit > 0 else 0
    current_page = (skip // limit) + 1 if limit > 0 else 1
    
    return DocumentListResponse(
        items=enhanced_items,
        total=total,
        page=current_page,
        size=limit,
        pages=pages
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    knowledge_base_id: int = Form(..., description="ID of the knowledge base"),
    file: UploadFile = File(..., description="Document file to upload"),
    metadata: str = Form("{}", description="Additional metadata as JSON string"),
    db: Session = Depends(get_db)
):
    """Upload a document to a knowledge base"""
    try:
        # Validate knowledge base
        kb_repo = KnowledgeBaseRepository(db)
        knowledge_base = kb_repo.get(knowledge_base_id)
        if not knowledge_base:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {knowledge_base_id} not found"
            )
        
        if not knowledge_base.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge base '{knowledge_base.name}' is not active"
            )
        
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{file_extension}' not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_path = Path(settings.upload_dir) / unique_filename
        
        # Save file
        file_content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(file_content)
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Create document record
        doc_repo = DocumentRepository(db)
        document_data = {
            "knowledge_base_id": knowledge_base_id,
            "filename": file.filename,
            "filepath": str(upload_path),
            "file_size": len(file_content),
            "mime_type": file.content_type,
            "status": DocumentStatus.UPLOADED,
            "title": metadata_dict.get("title", file.filename),
            "author": metadata_dict.get("author"),
            "language": metadata_dict.get("language", "en")
        }
        
        document = doc_repo.create(document_data)
        
        # Send to Kafka for processing
        try:
            kafka_producer = KafkaProducerService()
            message = {
                "document_id": document.id,
                "knowledge_base_id": knowledge_base_id,
                "filepath": str(upload_path),
                "filename": file.filename,
                "uploaded_at": datetime.now().isoformat()
            }
            kafka_producer.send_document_for_processing(message)
        except Exception as kafka_error:
            logger.warning(f"Failed to send to Kafka, document will be processed later: {kafka_error}")
            # We still return success, but document won't be processed immediately
        
        logger.info(f"Uploaded document: {file.filename} (ID: {document.id}) to knowledge base: {knowledge_base.name}")
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=file.filename,
            filepath=str(upload_path),
            status=DocumentStatus.UPLOADED,
            message="Document uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get a document by ID"""
    repo = DocumentRepository(db)
    kb_repo = KnowledgeBaseRepository(db)
    
    document = repo.get(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    # Prepare response
    document_dict = {**document.__dict__}
    document_dict.pop('_sa_instance_state', None)
    
    # Get knowledge base name
    kb_name = None
    if document.knowledge_base_id:
        kb = kb_repo.get(document.knowledge_base_id)
        if kb:
            kb_name = kb.name
    
    # Get chunk count
    chunk_count = 0  # Would need DocumentChunkRepository
    
    return DocumentResponse(
        **document_dict,
        knowledge_base_name=kb_name,
        chunk_count=chunk_count
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document"""
    repo = DocumentRepository(db)
    
    document = repo.get(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    try:
        # Delete physical file
        if os.path.exists(document.filepath):
            os.remove(document.filepath)
        
        # Delete from vector store (via service)
        from app.services.vector_store import VectorStoreService
        vector_store = VectorStoreService()
        vector_store.delete_by_document(document_id)
        
        # Delete from database
        repo.delete(document_id)
        
        logger.info(f"Deleted document: {document.filename} (ID: {document_id})")
        
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
    
    return None


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    db: Session = Depends(get_db)
):
    """Get chunks for a document"""
    repo = DocumentRepository(db)
    
    document = repo.get(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    # This would use DocumentChunkRepository
    # For now, return placeholder
    return {
        "document_id": document_id,
        "filename": document.filename,
        "chunks": [],
        "total": 0,
        "page": 1,
        "size": limit,
        "pages": 0
    }


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Reprocess a document"""
    repo = DocumentRepository(db)
    
    document = repo.get(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    # Update status to uploaded (will be picked up by worker)
    repo.update_status(document_id, DocumentStatus.UPLOADED)
    
    # Send to Kafka for processing
    try:
        kafka_producer = KafkaProducerService()
        message = {
            "document_id": document.id,
            "knowledge_base_id": document.knowledge_base_id,
            "filepath": document.filepath,
            "filename": document.filename,
            "reprocess": True,
            "requested_at": datetime.now().isoformat()
        }
        kafka_producer.send_document_for_processing(message)
    except Exception as kafka_error:
        logger.error(f"Failed to send to Kafka: {kafka_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue document for reprocessing"
        )
    
    logger.info(f"Queued document {document.filename} (ID: {document_id}) for reprocessing")
    
    return {
        "document_id": document_id,
        "filename": document.filename,
        "status": "queued_for_reprocessing",
        "message": "Document queued for reprocessing"
    }


@router.get("/stats/summary")
async def get_document_stats(
    knowledge_base_id: Optional[int] = Query(None, description="Filter by knowledge base ID"),
    db: Session = Depends(get_db)
):
    """Get document statistics"""
    repo = DocumentRepository(db)
    
    stats = repo.get_document_stats(knowledge_base_id)
    
    return {
        "knowledge_base_id": knowledge_base_id,
        "stats": stats,
        "timestamp": datetime.now().isoformat()
    }