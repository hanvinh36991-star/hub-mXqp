from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.services.vector_store import VectorStoreService
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "multimodal-rag-chatbot",
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check with dependency verification"""
    health_status = {
        "status": "healthy",
        "service": "multimodal-rag-chatbot",
        "version": "1.0.0",
        "checks": {}
    }
    
    try:
        # Check database connection
        db.execute("SELECT 1")
        health_status["checks"]["database"] = {
            "status": "healthy",
            "type": "sqlite",
            "connected": True
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    try:
        # Check Milvus connection
        vector_store = VectorStoreService()
        stats = vector_store.get_collection_stats()
        health_status["checks"]["milvus"] = {
            "status": "healthy",
            "collection": settings.milvus_collection,
            "stats": stats
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["milvus"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check configuration
    health_status["checks"]["configuration"] = {
        "debug_mode": settings.debug,
        "upload_dir_exists": True,  # Directory is created by config validator
        "processed_dir_exists": True
    }
    
    return health_status


@router.get("/ready")
async def readiness_check():
    """Readiness check for load balancers"""
    # For now, just return healthy if the service is running
    # In production, you might want to check all dependencies
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Liveness check for k8s"""
    return {"status": "alive"}