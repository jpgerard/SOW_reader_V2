"""Vector-based semantic search for proposal matching."""

import numpy as np
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
import re
import logging
import hashlib
import os
import pickle
from pathlib import Path

# Configure production logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    section_id: str
    text: str
    start_char: int
    end_char: int

@dataclass
class SearchResult:
    """Result from a vector search."""
    chunk: ChunkMetadata
    similarity_score: float

class VectorSearchError(Exception):
    """Base exception for vector search errors."""
    pass

class ProposalVectorizer:
    """Handles text vectorization and semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a specific model."""
        try:
            logger.info(f"Initializing vectorizer with model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.index = None
            self.chunks = []
            
            # Setup cache in app data directory
            app_data = os.getenv('APPDATA') or os.path.expanduser('~/.local/share')
            self.cache_dir = os.path.join(app_data, 'sow_reader', 'embeddings_cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache directory initialized at: {self.cache_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorizer: {str(e)}")
            raise VectorSearchError(f"Vectorizer initialization failed: {str(e)}")
        
    def _get_cache_path(self, text: str) -> str:
        """Get cache file path for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
        
    def _create_chunks(self, text: str, max_length: int = 512) -> List[ChunkMetadata]:
        """Split text into chunks, preserving section structure."""
        chunks = []
        
        try:
            # Find sections with X.X.X format
            section_pattern = r'(?:(?<=\n)|^)\s*(\d+\.\d+\.\d+)\s*(.*?)(?=\n\s*\d+\.\d+\.\d+|$)'
            
            matches = list(re.finditer(section_pattern, text, re.DOTALL))
            logger.info(f"Found {len(matches)} sections to process")
            
            for match in matches:
                section_id = match.group(1)
                content = match.group(2).strip()
                
                # Always create at least one chunk per section
                if not content:
                    chunks.append(ChunkMetadata(
                        section_id=section_id,
                        text=f"{section_id}",
                        start_char=match.start(1),
                        end_char=match.end(1)
                    ))
                    continue
                    
                # Create a single chunk per section unless it's too long
                if len(content) <= max_length:
                    chunks.append(ChunkMetadata(
                        section_id=section_id,
                        text=f"{section_id} {content}",
                        start_char=match.start(),
                        end_char=match.end()
                    ))
                else:
                    # Split into sentences for long sections
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    current_chunk = ""
                    chunk_start = match.start(2)
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > max_length and current_chunk:
                            chunks.append(ChunkMetadata(
                                section_id=section_id,
                                text=f"{section_id} {current_chunk.strip()}",
                                start_char=chunk_start,
                                end_char=chunk_start + len(current_chunk)
                            ))
                            current_chunk = sentence
                            chunk_start = chunk_start + len(current_chunk)
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
                    
                    if current_chunk:
                        chunks.append(ChunkMetadata(
                            section_id=section_id,
                            text=f"{section_id} {current_chunk.strip()}",
                            start_char=chunk_start,
                            end_char=chunk_start + len(current_chunk)
                        ))
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise VectorSearchError(f"Failed to create text chunks: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        cache_path = self._get_cache_path(text)
        
        try:
            # Try to load from cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {str(e)}")
            
            # Generate new embedding
            # Disable gradients and JIT for inference
            with torch.no_grad(), torch.jit.optimized_execution(False):
                embedding = self.model.encode(text, convert_to_tensor=True)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                embedding_np = embedding.cpu().numpy()
            
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding_np, f)
            except Exception as e:
                logger.warning(f"Failed to save to cache: {str(e)}")
                
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise VectorSearchError(f"Failed to generate embedding: {str(e)}")

    def index_proposal(self, proposal_text: str):
        """Create searchable index from proposal text."""
        try:
            # Create chunks
            self.chunks = self._create_chunks(proposal_text)
            if not self.chunks:
                logger.warning("No chunks were created from the proposal text")
                return
                
            # Generate embeddings
            embeddings = []
            for chunk in self.chunks:
                embedding = self._get_embedding(chunk.text)
                embeddings.append(embedding)
                
            # Stack embeddings
            embeddings_array = np.vstack(embeddings)
            dimension = embeddings_array.shape[1]
            
            # Initialize FAISS index for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            normalized_embeddings = embeddings_array.astype('float32')
            self.index.add(normalized_embeddings)
            logger.info(f"Created search index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error indexing proposal: {str(e)}")
            raise VectorSearchError(f"Failed to index proposal: {str(e)}")

    def search(self, requirement_text: str, top_k: int = 3) -> List[SearchResult]:
        """Search for most similar proposal chunks to a requirement."""
        try:
            if not self.index:
                raise VectorSearchError("No index exists. Call index_proposal first.")
                
            if not self.chunks:
                logger.warning("No chunks available for searching")
                return []
                
            # Get requirement embedding
            query_embedding = self._get_embedding(requirement_text)
            
            # Search index
            scores, indices = self.index.search(
                query_embedding.astype('float32').reshape(1, -1),
                min(top_k, len(self.chunks))
            )
            
            # Convert to search results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Convert cosine similarity to 0-1 range
                similarity = (score + 1) / 2
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    similarity_score=float(similarity)
                ))
                logger.info(f"Match found: section={self.chunks[idx].section_id}, score={similarity:.3f}")
            
            return results
            
        except VectorSearchError:
            raise
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise VectorSearchError(f"Search operation failed: {str(e)}")
