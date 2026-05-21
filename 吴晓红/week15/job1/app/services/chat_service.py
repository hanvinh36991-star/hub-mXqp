from typing import List, Dict, Any, Optional, Tuple
import uuid
import time
import json
from loguru import logger
from app.core.config import settings
from app.services.models import ModelService
from app.services.vector_store import VectorStoreService
from app.repositories.chat import ChatSessionRepository, ChatMessageRepository
from app.repositories.document import DocumentRepository
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole


class ChatService:
    """Service for handling chat conversations with RAG"""
    
    def __init__(
        self,
        model_service: Optional[ModelService] = None,
        vector_store: Optional[VectorStoreService] = None,
        session_repo: Optional[ChatSessionRepository] = None,
        message_repo: Optional[ChatMessageRepository] = None,
        document_repo: Optional[DocumentRepository] = None
    ):
        self.model_service = model_service or ModelService()
        self.vector_store = vector_store or VectorStoreService()
        self.session_repo = session_repo
        self.message_repo = message_repo
        self.document_repo = document_repo
    
    async def process_chat_request(
        self,
        request: ChatRequest,
        db = None
    ) -> ChatResponse:
        """Process a chat request with RAG"""
        start_time = time.time()
        
        try:
            # Get or create session
            session = await self._get_or_create_session(
                request.session_id,
                request.knowledge_base_id,
                db
            )
            
            # Save user message
            user_message = await self._save_message(
                session_id=session.id,
                role=MessageRole.USER,
                content=request.message,
                db=db
            )
            
            # Retrieve relevant chunks
            retrieved_chunks = await self._retrieve_chunks(
                query=request.message,
                knowledge_base_id=request.knowledge_base_id,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold
            )
            
            # Generate response
            response_text = await self._generate_response(
                question=request.message,
                retrieved_chunks=retrieved_chunks,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Save assistant message
            assistant_message = await self._save_message(
                session_id=session.id,
                role=MessageRole.ASSISTANT,
                content=response_text,
                retrieved_chunk_ids=json.dumps([c.get('chunk_id') for c in retrieved_chunks]),
                model_used=request.model or settings.qwen_model,
                tokens_used=len(response_text.split()),  # Approximate token count
                db=db
            )
            
            # Get source document names
            source_documents = await self._get_source_documents(retrieved_chunks, db)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = ChatResponse(
                session_id=session.session_id,
                message_id=assistant_message.id,
                response=response_text,
                model_used=request.model or settings.qwen_model,
                tokens_used=len(response_text.split()),
                processing_time=processing_time,
                retrieved_chunks=retrieved_chunks,
                retrieval_scores=[c.get('score', 0.0) for c in retrieved_chunks],
                source_documents=source_documents
            )
            
            logger.info(f"Processed chat request in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process chat request: {e}")
            raise
    
    async def _get_or_create_session(
        self,
        session_id: Optional[str],
        knowledge_base_id: int,
        db
    ) -> Dict[str, Any]:
        """Get existing session or create new one"""
        if session_id:
            # Try to get existing session
            if self.session_repo:
                session = self.session_repo.get_by_session_id(session_id)
                if session:
                    return session
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        session_title = f"Chat about knowledge base {knowledge_base_id}"
        
        if self.session_repo and db:
            session_data = {
                "session_id": new_session_id,
                "knowledge_base_id": knowledge_base_id,
                "title": session_title
            }
            session = self.session_repo.create(session_data)
        else:
            # Fallback to dict if repos not available
            session = {
                "id": 1,
                "session_id": new_session_id,
                "knowledge_base_id": knowledge_base_id,
                "title": session_title
            }
        
        return session
    
    async def _save_message(
        self,
        session_id: int,
        role: MessageRole,
        content: str,
        retrieved_chunk_ids: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        db = None
    ) -> Dict[str, Any]:
        """Save chat message to database"""
        if self.message_repo and db:
            message_data = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "model_used": model_used,
                "tokens_used": tokens_used
            }
            message = self.message_repo.create(message_data)
        else:
            # Fallback to dict if repos not available
            message = {
                "id": 1,
                "session_id": session_id,
                "role": role,
                "content": content,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "model_used": model_used,
                "tokens_used": tokens_used
            }
        
        return message
    
    async def _retrieve_chunks(
        self,
        query: str,
        knowledge_base_id: int,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using multimodal search"""
        try:
            # Get query embedding
            query_embedding = self.model_service.get_text_embedding(query).tolist()
            
            # Search using BGE vectors (text search)
            text_results = self.vector_store.search(
                query_vector=query_embedding,
                vector_field="bge_vector",
                knowledge_base_id=knowledge_base_id,
                top_k=top_k
            )
            
            # Also try CLIP text vectors for multimodal queries
            clip_query_embedding = self.model_service.get_clip_text_embedding(query).tolist()
            clip_results = self.vector_store.search(
                query_vector=clip_query_embedding,
                vector_field="clip_text_vector",
                knowledge_base_id=knowledge_base_id,
                top_k=top_k
            )
            
            # Combine and deduplicate results
            all_results = {}
            for result in text_results + clip_results:
                chunk_id = result.get('chunk_id')
                if chunk_id not in all_results or result.get('score', 0) > all_results[chunk_id].get('score', 0):
                    all_results[chunk_id] = result
            
            # Filter by similarity threshold and sort by score
            filtered_results = [
                result for result in all_results.values()
                if result.get('score', 0) >= similarity_threshold
            ]
            filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Take top-k results
            final_results = filtered_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} chunks for query")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return []
    
    async def _generate_response(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """Generate response using retrieved chunks"""
        try:
            # Check if we have any retrieved chunks
            if not retrieved_chunks:
                return "I couldn't find relevant information in the knowledge base to answer your question."
            
            # Create RAG prompt
            prompt = self.model_service.create_rag_prompt(question, retrieved_chunks)
            
            # Prepare messages for the model
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided information."},
                {"role": "user", "content": prompt}
            ]
            
            # Generate response
            response = self.model_service.generate_response(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I encountered an error while generating a response. Please try again."
    
    async def _get_source_documents(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        db = None
    ) -> List[str]:
        """Get source document names for retrieved chunks"""
        try:
            if not self.document_repo or not db:
                return ["Unknown source"]
            
            document_ids = set()
            for chunk in retrieved_chunks:
                doc_id = chunk.get('document_id')
                if doc_id:
                    document_ids.add(doc_id)
            
            source_documents = []
            for doc_id in document_ids:
                document = self.document_repo.get(doc_id)
                if document:
                    source_documents.append(document.filename)
            
            return source_documents if source_documents else ["Unknown source"]
            
        except Exception as e:
            logger.error(f"Failed to get source documents: {e}")
            return ["Unknown source"]
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 50,
        db = None
    ) -> Dict[str, Any]:
        """Get chat history for a session"""
        try:
            if not self.session_repo or not self.message_repo or not db:
                return {"session": None, "messages": []}
            
            # Get session
            session = self.session_repo.get_by_session_id(session_id)
            if not session:
                return {"session": None, "messages": []}
            
            # Get messages
            messages = self.message_repo.get_by_session(session.id, limit=limit)
            
            return {
                "session": session,
                "messages": messages
            }
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            raise