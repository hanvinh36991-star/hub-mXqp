from fastapi import APIRouter
from app.api.endpoints import knowledge_bases, documents, chat, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(knowledge_bases.router, prefix="/knowledge-bases", tags=["knowledge-bases"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])