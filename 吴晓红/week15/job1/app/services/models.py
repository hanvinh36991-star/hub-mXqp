from typing import List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from loguru import logger
from PIL import Image
import requests
from io import BytesIO
from app.core.config import settings


class ModelService:
    """Service for AI model operations"""
    
    def __init__(self):
        self.bge_model = None
        self.clip_model = None
        self.qwen_client = None
        self._load_models()
    
    def _load_models(self):
        """Load models based on configuration"""
        # Load BGE model for text embeddings
        if settings.bge_model_path:
            try:
                self.bge_model = SentenceTransformer(settings.bge_model_path)
                logger.info(f"Loaded BGE model from {settings.bge_model_path}")
            except Exception as e:
                logger.error(f"Failed to load BGE model: {e}")
        
        # Load CLIP model for multimodal embeddings
        if settings.clip_model_path:
            try:
                self.clip_model = SentenceTransformer(
                    settings.clip_model_path,
                    trust_remote_code=True,
                    truncate_dim=1024
                )
                logger.info(f"Loaded CLIP model from {settings.clip_model_path}")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
        
        # Initialize Qwen client for generation
        if settings.qwen_api_key:
            try:
                self.qwen_client = OpenAI(
                    api_key=settings.qwen_api_key,
                    base_url=settings.qwen_base_url
                )
                logger.info("Initialized Qwen client")
            except Exception as e:
                logger.error(f"Failed to initialize Qwen client: {e}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using BGE model"""
        if not self.bge_model:
            raise ValueError("BGE model not loaded")
        
        try:
            embedding = self.bge_model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get text embedding: {e}")
            raise
    
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using CLIP model"""
        if not self.clip_model:
            raise ValueError("CLIP model not loaded")
        
        try:
            embedding = self.clip_model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get CLIP text embedding: {e}")
            raise
    
    def get_clip_image_embedding(self, image_path: str) -> np.ndarray:
        """Get image embedding using CLIP model"""
        if not self.clip_model:
            raise ValueError("CLIP model not loaded")
        
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            # Encode image
            embedding = self.clip_model.encode(image, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get image embedding for {image_path}: {e}")
            raise
    
    def generate_response(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Union[str, dict]:
        """Generate response using Qwen model"""
        if not self.qwen_client:
            raise ValueError("Qwen client not initialized")
        
        try:
            model = model or settings.qwen_model
            
            response = self.qwen_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def batch_encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Batch encode multiple texts"""
        if not self.bge_model:
            raise ValueError("BGE model not loaded")
        
        try:
            embeddings = self.bge_model.encode(texts, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to batch encode texts: {e}")
            raise
    
    def extract_text_from_markdown(self, markdown_text: str) -> Tuple[str, List[str]]:
        """
        Extract text and image references from markdown
        
        Returns:
            Tuple of (text_content, image_paths)
        """
        lines = markdown_text.split('\n')
        text_lines = []
        image_paths = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('!['):
                # Extract image path from markdown image syntax ![alt](path)
                parts = line.split('](')
                if len(parts) > 1:
                    image_path = parts[1].rstrip(')')
                    image_paths.append(image_path)
            elif line:
                text_lines.append(line)
        
        text_content = '\n'.join(text_lines)
        return text_content, image_paths
    
    def create_rag_prompt(
        self,
        question: str,
        retrieved_chunks: List[dict],
        include_images: bool = True
    ) -> str:
        """Create prompt for RAG response generation"""
        # Format retrieved content
        formatted_chunks = []
        for chunk in retrieved_chunks:
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            source_info = metadata.get('source', 'Unknown')
            
            chunk_text = f"Source: {source_info}\nContent: {content}\n"
            formatted_chunks.append(chunk_text)
        
        # Build prompt
        prompt = f"""Based on the following retrieved information, answer the user's question.

User Question: {question}

Retrieved Information:
{"=" * 50}
{"".join(formatted_chunks)}
{"=" * 50}

Please provide a comprehensive answer based on the retrieved information. If the information includes image references, describe what the images show when relevant to the answer.

Answer:"""
        
        return prompt