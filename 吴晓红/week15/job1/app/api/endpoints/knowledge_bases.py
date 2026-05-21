from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    KnowledgeBaseListResponse
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    active_only: bool = Query(True, description="Only return active knowledge bases"),
    search: Optional[str] = Query(None, description="Search query for name or description"),
    db: Session = Depends(get_db)
):
    """List knowledge bases"""
    repo = KnowledgeBaseRepository(db)
    
    if search:
        items = repo.search(query=search, skip=skip, limit=limit)
        total = repo.count({"is_active": True} if active_only else {})
    elif active_only:
        items = repo.get_active_bases(skip=skip, limit=limit)
        total = repo.count({"is_active": True})
    else:
        items = repo.get_multi(skip=skip, limit=limit)
        total = repo.count()
    
    # Enhance items with document counts
    enhanced_items = []
    for item in items:
        item_dict = {**item.__dict__}
        # Remove SQLAlchemy internal attribute
        item_dict.pop('_sa_instance_state', None)
        
        # Add document counts
        item_dict["document_count"] = repo.get_document_count(item.id)
        item_dict["processed_document_count"] = repo.get_processed_document_count(item.id)
        
        enhanced_items.append(KnowledgeBaseResponse(**item_dict))
    
    pages = (total + limit - 1) // limit if limit > 0 else 0
    current_page = (skip // limit) + 1 if limit > 0 else 1
    
    return KnowledgeBaseListResponse(
        items=enhanced_items,
        total=total,
        page=current_page,
        size=limit,
        pages=pages
    )


@router.post("/", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    knowledge_base: KnowledgeBaseCreate,
    db: Session = Depends(get_db)
):
    """Create a new knowledge base"""
    repo = KnowledgeBaseRepository(db)
    
    # Check if name already exists
    existing = repo.get_by_name(knowledge_base.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Knowledge base with name '{knowledge_base.name}' already exists"
        )
    
    # Create knowledge base
    knowledge_base_data = knowledge_base.dict()
    created_kb = repo.create(knowledge_base_data)
    
    # Prepare response
    response_data = {**created_kb.__dict__}
    response_data.pop('_sa_instance_state', None)
    response_data["document_count"] = 0
    response_data["processed_document_count"] = 0
    
    logger.info(f"Created knowledge base: {created_kb.name} (ID: {created_kb.id})")
    return KnowledgeBaseResponse(**response_data)


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    knowledge_base_id: int,
    db: Session = Depends(get_db)
):
    """Get a knowledge base by ID"""
    repo = KnowledgeBaseRepository(db)
    
    knowledge_base = repo.get(knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base with ID {knowledge_base_id} not found"
        )
    
    # Prepare response
    response_data = {**knowledge_base.__dict__}
    response_data.pop('_sa_instance_state', None)
    response_data["document_count"] = repo.get_document_count(knowledge_base_id)
    response_data["processed_document_count"] = repo.get_processed_document_count(knowledge_base_id)
    
    return KnowledgeBaseResponse(**response_data)


@router.put("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    knowledge_base_id: int,
    knowledge_base_update: KnowledgeBaseUpdate,
    db: Session = Depends(get_db)
):
    """Update a knowledge base"""
    repo = KnowledgeBaseRepository(db)
    
    # Get existing knowledge base
    existing_kb = repo.get(knowledge_base_id)
    if not existing_kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base with ID {knowledge_base_id} not found"
        )
    
    # Check if new name conflicts with existing knowledge base
    if knowledge_base_update.name and knowledge_base_update.name != existing_kb.name:
        conflicting_kb = repo.get_by_name(knowledge_base_update.name)
        if conflicting_kb and conflicting_kb.id != knowledge_base_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge base with name '{knowledge_base_update.name}' already exists"
            )
    
    # Update knowledge base
    update_data = knowledge_base_update.dict(exclude_unset=True)
    updated_kb = repo.update(existing_kb, update_data)
    
    # Prepare response
    response_data = {**updated_kb.__dict__}
    response_data.pop('_sa_instance_state', None)
    response_data["document_count"] = repo.get_document_count(knowledge_base_id)
    response_data["processed_document_count"] = repo.get_processed_document_count(knowledge_base_id)
    
    logger.info(f"Updated knowledge base: {updated_kb.name} (ID: {knowledge_base_id})")
    return KnowledgeBaseResponse(**response_data)


@router.delete("/{knowledge_base_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    knowledge_base_id: int,
    db: Session = Depends(get_db)
):
    """Delete a knowledge base (soft delete by marking as inactive)"""
    repo = KnowledgeBaseRepository(db)
    
    knowledge_base = repo.get(knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base with ID {knowledge_base_id} not found"
        )
    
    # Soft delete by marking as inactive
    update_data = {"is_active": False}
    repo.update(knowledge_base, update_data)
    
    logger.info(f"Deleted knowledge base: {knowledge_base.name} (ID: {knowledge_base_id})")
    return None


@router.get("/{knowledge_base_id}/stats")
async def get_knowledge_base_stats(
    knowledge_base_id: int,
    db: Session = Depends(get_db)
):
    """Get statistics for a knowledge base"""
    repo = KnowledgeBaseRepository(db)
    
    knowledge_base = repo.get(knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base with ID {knowledge_base_id} not found"
        )
    
    document_count = repo.get_document_count(knowledge_base_id)
    processed_document_count = repo.get_processed_document_count(knowledge_base_id)
    
    return {
        "knowledge_base_id": knowledge_base_id,
        "name": knowledge_base.name,
        "document_count": document_count,
        "processed_document_count": processed_document_count,
        "pending_document_count": document_count - processed_document_count,
        "is_active": knowledge_base.is_active,
        "created_at": knowledge_base.created_at,
        "updated_at": knowledge_base.updated_at
    }