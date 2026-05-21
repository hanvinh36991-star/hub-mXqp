from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.database import ChatSession, ChatMessage
from app.repositories.base import BaseRepository
from app.schemas.chat import MessageRole


class ChatSessionRepository(BaseRepository[ChatSession, Dict, Dict]):
    """Repository for chat session operations"""
    
    def __init__(self, db: Session):
        super().__init__(ChatSession, db)
    
    def get_by_session_id(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by session ID"""
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
    
    def get_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> List[ChatSession]:
        """Get chat sessions for a user"""
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_knowledge_base(self, knowledge_base_id: int, skip: int = 0, limit: int = 100) -> List[ChatSession]:
        """Get chat sessions for a knowledge base"""
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.knowledge_base_id == knowledge_base_id)
            .order_by(desc(ChatSession.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_session_title(self, session_id: str, title: str) -> Optional[ChatSession]:
        """Update session title"""
        session = self.get_by_session_id(session_id)
        if session:
            session.title = title
            self.db.commit()
            self.db.refresh(session)
        return session


class ChatMessageRepository(BaseRepository[ChatMessage, Dict, Dict]):
    """Repository for chat message operations"""
    
    def __init__(self, db: Session):
        super().__init__(ChatMessage, db)
    
    def get_by_session(self, session_id: int, limit: int = 100) -> List[ChatMessage]:
        """Get messages for a session"""
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .all()
        )
    
    def get_latest_by_session(self, session_id: int, limit: int = 10) -> List[ChatMessage]:
        """Get latest messages for a session"""
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(limit)
            .all()
        )
    
    def get_by_role(self, session_id: int, role: MessageRole, limit: int = 100) -> List[ChatMessage]:
        """Get messages by role in a session"""
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id, ChatMessage.role == role)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .all()
        )
    
    def count_by_session(self, session_id: int) -> int:
        """Count messages in a session"""
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .count()
        )
    
    def delete_by_session(self, session_id: int) -> int:
        """Delete all messages in a session"""
        count = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .delete()
        )
        self.db.commit()
        return count