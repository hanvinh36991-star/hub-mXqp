from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import asyncio
from app.models.database import get_db
from app.repositories.chat import ChatSessionRepository, ChatMessageRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryResponse
from app.services.chat_service import ChatService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Process a chat query with RAG"""
    try:
        # Validate knowledge base
        kb_repo = KnowledgeBaseRepository(db)
        knowledge_base = kb_repo.get(request.knowledge_base_id)
        if not knowledge_base:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {request.knowledge_base_id} not found"
            )
        
        if not knowledge_base.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge base '{knowledge_base.name}' is not active"
            )
        
        # Initialize services
        session_repo = ChatSessionRepository(db)
        message_repo = ChatMessageRepository(db)
        
        chat_service = ChatService(
            session_repo=session_repo,
            message_repo=message_repo,
            document_repo=None  # Would need DocumentRepository if we want source document names
        )
        
        # Process chat request
        response = await chat_service.process_chat_request(request, db)
        
        logger.info(f"Processed chat query for session {response.session_id}, "
                   f"knowledge base {request.knowledge_base_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process chat query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}"
        )


@router.post("/query/stream")
async def chat_query_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Process a chat query with streaming response"""
    # This would implement streaming response
    # For now, return a placeholder response
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Streaming is not yet implemented"
    )


@router.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of messages to return"),
    db: Session = Depends(get_db)
):
    """Get chat history for a session"""
    try:
        session_repo = ChatSessionRepository(db)
        message_repo = ChatMessageRepository(db)
        
        # Get session
        session = session_repo.get_by_session_id(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with ID {session_id} not found"
            )
        
        # Get messages
        messages = message_repo.get_by_session(session.id, limit=limit)
        
        # Convert session to response format
        session_dict = {**session.__dict__}
        session_dict.pop('_sa_instance_state', None)
        session_dict["message_count"] = message_repo.count_by_session(session.id)
        
        # Convert messages to response format
        message_list = []
        for msg in messages:
            msg_dict = {**msg.__dict__}
            msg_dict.pop('_sa_instance_state', None)
            message_list.append(msg_dict)
        
        return ChatHistoryResponse(
            session=session_dict,
            messages=message_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat history: {str(e)}"
        )


@router.get("/sessions")
async def list_chat_sessions(
    knowledge_base_id: Optional[int] = Query(None, description="Filter by knowledge base ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    db: Session = Depends(get_db)
):
    """List chat sessions"""
    try:
        session_repo = ChatSessionRepository(db)
        message_repo = ChatMessageRepository(db)
        
        if knowledge_base_id:
            sessions = session_repo.get_by_knowledge_base(knowledge_base_id, skip=skip, limit=limit)
            total = session_repo.count({"knowledge_base_id": knowledge_base_id})
        elif user_id:
            sessions = session_repo.get_by_user(user_id, skip=skip, limit=limit)
            total = session_repo.count({"user_id": user_id})
        else:
            sessions = session_repo.get_multi(skip=skip, limit=limit)
            total = session_repo.count()
        
        # Enhance sessions with message counts
        enhanced_sessions = []
        for session in sessions:
            session_dict = {**session.__dict__}
            session_dict.pop('_sa_instance_state', None)
            session_dict["message_count"] = message_repo.count_by_session(session.id)
            enhanced_sessions.append(session_dict)
        
        pages = (total + limit - 1) // limit if limit > 0 else 0
        current_page = (skip // limit) + 1 if limit > 0 else 1
        
        return {
            "sessions": enhanced_sessions,
            "total": total,
            "page": current_page,
            "size": limit,
            "pages": pages
        }
        
    except Exception as e:
        logger.error(f"Failed to list chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list chat sessions: {str(e)}"
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Delete a chat session and all its messages"""
    try:
        session_repo = ChatSessionRepository(db)
        message_repo = ChatMessageRepository(db)
        
        session = session_repo.get_by_session_id(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with ID {session_id} not found"
            )
        
        # Delete messages first
        deleted_messages = message_repo.delete_by_session(session.id)
        
        # Delete session
        session_repo.delete(session.id)
        
        logger.info(f"Deleted chat session {session_id} with {deleted_messages} messages")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat session: {str(e)}"
        )


@router.put("/sessions/{session_id}/title")
async def update_chat_session_title(
    session_id: str,
    title: str,
    db: Session = Depends(get_db)
):
    """Update chat session title"""
    try:
        session_repo = ChatSessionRepository(db)
        
        session = session_repo.update_session_title(session_id, title)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with ID {session_id} not found"
            )
        
        return {
            "session_id": session_id,
            "title": session.title,
            "updated_at": session.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update chat session title: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update chat session title: {str(e)}"
        )


@router.get("/knowledge-bases/{knowledge_base_id}/suggestions")
async def get_query_suggestions(
    knowledge_base_id: int,
    db: Session = Depends(get_db)
):
    """Get query suggestions for a knowledge base"""
    # This would analyze the knowledge base content and provide
    # suggested questions or topics
    # For now, return placeholder
    
    return {
        "knowledge_base_id": knowledge_base_id,
        "suggestions": [
            "What are the main topics covered in these documents?",
            "Can you summarize the key points?",
            "Are there any images or diagrams that explain important concepts?",
            "What recent updates or changes are mentioned?"
        ],
        "popular_topics": ["overview", "key_concepts", "faq", "troubleshooting"]
    }