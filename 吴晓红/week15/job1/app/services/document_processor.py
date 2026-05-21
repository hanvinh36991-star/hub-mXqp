import os
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from pathlib import Path
from loguru import logger
from app.core.config import settings
from app.services.models import ModelService


class DocumentProcessor:
    """Service for processing documents (PDF parsing, chunking, embedding)"""
    
    def __init__(self, model_service: Optional[ModelService] = None):
        self.model_service = model_service or ModelService()
        self.upload_dir = Path(settings.upload_dir)
        self.processed_dir = Path(settings.processed_dir)
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_with_mineru(self, pdf_path: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Parse PDF using Mineru document parser
        
        Returns:
            Path to the output directory containing markdown and images
        """
        try:
            if output_dir is None:
                # Create output directory based on PDF filename
                pdf_stem = pdf_path.stem
                output_dir = self.processed_dir / pdf_stem
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run mineru command
            cmd = [
                "mineru",
                "-p", str(pdf_path),
                "-o", str(output_dir),
                "-b", "vlm-http-client",
                "-u", settings.mineru_endpoint
            ]
            
            logger.info(f"Running mineru: {' '.join(cmd)}")
            
            # Execute command with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Mineru failed: {result.stderr}")
                raise RuntimeError(f"Mineru parsing failed: {result.stderr}")
            
            logger.info(f"Mineru completed: {output_dir}")
            return output_dir
            
        except subprocess.TimeoutExpired:
            logger.error("Mineru timeout after 10 minutes")
            raise RuntimeError("Document parsing timeout")
        except Exception as e:
            logger.error(f"Failed to parse document with mineru: {e}")
            raise
    
    def split_markdown_by_headers(self, markdown_path: Path, max_length: int = 1024) -> List[Dict[str, Any]]:
        """
        Split markdown document by headers
        
        Args:
            markdown_path: Path to markdown file
            max_length: Maximum chunk length in characters
            
        Returns:
            List of chunks with metadata
        """
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            current_chunk = []
            current_header = "Document"
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if line is a header (starts with #)
                if line_stripped.startswith('#'):
                    # Save current chunk if it exists
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk).strip()
                        if chunk_text:
                            chunks.append({
                                'text': chunk_text,
                                'header': current_header,
                                'path': str(markdown_path),
                                'has_images': any('![' in line for line in current_chunk)
                            })
                    
                    # Start new chunk
                    current_chunk = [line]
                    current_header = line_stripped
                else:
                    current_chunk.append(line)
            
            # Add the last chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'header': current_header,
                        'path': str(markdown_path),
                        'has_images': any('![' in line for line in current_chunk)
                    })
            
            # Further split large chunks
            final_chunks = []
            for chunk in chunks:
                text = chunk['text']
                header = chunk['header']
                
                if len(text) <= max_length:
                    final_chunks.append(chunk)
                else:
                    # Split by sentences or paragraphs
                    paragraphs = text.split('\n\n')
                    current_paragraphs = []
                    current_length = 0
                    
                    for para in paragraphs:
                        para_length = len(para)
                        
                        if current_length + para_length <= max_length:
                            current_paragraphs.append(para)
                            current_length += para_length
                        else:
                            # Save current chunk
                            if current_paragraphs:
                                chunk_text = '\n\n'.join(current_paragraphs)
                                final_chunks.append({
                                    'text': chunk_text,
                                    'header': f"{header} (continued)",
                                    'path': str(markdown_path),
                                    'has_images': any('![' in para for para in current_paragraphs)
                                })
                            
                            # Start new chunk with current paragraph
                            current_paragraphs = [para]
                            current_length = para_length
                    
                    # Add remaining paragraphs
                    if current_paragraphs:
                        chunk_text = '\n\n'.join(current_paragraphs)
                        final_chunks.append({
                            'text': chunk_text,
                            'header': f"{header} (continued)",
                            'path': str(markdown_path),
                            'has_images': any('![' in para for para in current_paragraphs)
                        })
            
            logger.info(f"Split markdown into {len(final_chunks)} chunks")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Failed to split markdown: {e}")
            raise
    
    def extract_images_from_markdown(self, markdown_path: Path) -> List[str]:
        """Extract image paths from markdown file"""
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            image_paths = []
            lines = content.split('\n')
            
            for line in lines:
                if line.strip().startswith('!['):
                    # Extract path from ![alt](path)
                    parts = line.split('](')
                    if len(parts) > 1:
                        image_path = parts[1].rstrip(')')
                        
                        # Convert to absolute path if relative
                        if not os.path.isabs(image_path):
                            markdown_dir = markdown_path.parent
                            image_path = str((markdown_dir / image_path).resolve())
                        
                        image_paths.append(image_path)
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Failed to extract images from markdown: {e}")
            return []
    
    def process_chunk(
        self,
        chunk: Dict[str, Any],
        document_id: int,
        knowledge_base_id: int,
        chunk_index: int
    ) -> Dict[str, Any]:
        """Process a single chunk and generate embeddings"""
        try:
            text = chunk.get('text', '')
            has_images = chunk.get('has_images', False)
            
            # Extract text and images
            text_content, image_paths = self.model_service.extract_text_from_markdown(text)
            
            # Generate embeddings
            bge_embedding = self.model_service.get_text_embedding(text_content)
            clip_text_embedding = self.model_service.get_clip_text_embedding(text_content)
            
            # Generate image embedding if there are images
            clip_image_embedding = None
            if image_paths and has_images:
                try:
                    # Use first image for embedding
                    first_image = image_paths[0]
                    clip_image_embedding = self.model_service.get_clip_image_embedding(first_image)
                except Exception as e:
                    logger.warning(f"Failed to get image embedding: {e}")
                    clip_image_embedding = np.zeros(1024)
            
            # Prepare chunk data for vector store
            chunk_data = {
                'document_id': document_id,
                'chunk_id': chunk_index,
                'knowledge_base_id': knowledge_base_id,
                'content': text,
                'content_type': 'mixed' if has_images else 'text',
                'bge_vector': bge_embedding.tolist(),
                'clip_text_vector': clip_text_embedding.tolist(),
                'clip_image_vector': clip_image_embedding.tolist() if clip_image_embedding is not None else [0.0] * 1024,
                'metadata': {
                    'header': chunk.get('header', ''),
                    'path': chunk.get('path', ''),
                    'has_images': has_images,
                    'image_paths': image_paths,
                    'text_length': len(text),
                    'source': f"Document {document_id}, Chunk {chunk_index}"
                }
            }
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_index}: {e}")
            raise
    
    def find_markdown_files(self, directory: Path) -> List[Path]:
        """Find all markdown files in a directory"""
        markdown_files = list(directory.rglob("*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files in {directory}")
        return markdown_files