from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
from loguru import logger
import sys
from app.core.config import settings
from app.api.api import api_router
from app.models.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Intercept standard logging with loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

logging.getLogger().handlers = [InterceptHandler()]

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A Multimodal RAG Chatbot for processing and querying documents with text and images",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# Add middleware
if settings.debug:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]  # Update with your domains
    )

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    logger.info(f"Server running on {settings.host}:{settings.port}")
    logger.info(f"Debug mode: {settings.debug}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else None,
        "api_version": "v1",
        "endpoints": {
            "health": "/api/v1/health",
            "knowledge_bases": "/api/v1/knowledge-bases",
            "documents": "/api/v1/documents",
            "chat": "/api/v1/chat"
        }
    }


@app.get("/version")
async def version():
    """Version endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "api_version": "v1"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )