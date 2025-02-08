"""Enhanced vector search implementation optimized for Streamlit Cloud."""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import faiss
import torch

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class EnhancedVectorizer:
    """Enhanced vector search with CPU optimization for Streamlit Cloud."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize the vectorizer.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            logger.info(f"Initializing vectorizer with model: {model_name}")
            # Force CPU usage for Streamlit Cloud compatibility
            self.device = torch.device("cpu")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index (CPU version)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.doc_ids = []
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorizer: {str(e)}")
            raise
            
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding
        """
        try:
            # Normalize text and generate embedding
            text = text.strip().replace('\n', ' ')
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.reshape(1, -1).astype('float32')
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
            
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the search index.
        
        Args:
            documents: List of documents with text content and metadata
        """
        try:
            for doc in documents:
                # Generate embedding
                embedding = self._get_embedding(doc['text'])
                
                # Add to index
                self.index.add(embedding)
                self.documents.append(doc)
                self.doc_ids.append(doc.get('id', str(len(self.doc_ids))))
                
            logger.info(f"Added {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
            
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if not self.documents:
                return []
                
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Search index
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):  # Ensure valid index
                    doc = self.documents[idx]
                    results.append(SearchResult(
                        id=self.doc_ids[idx],
                        content=doc['text'],
                        score=float(score),
                        metadata=doc.get('metadata', {})
                    ))
                    
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
            
    def get_score(self, doc_id: str) -> float:
        """Get the score for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Score between 0 and 1
        """
        try:
            if doc_id in self.doc_ids:
                idx = self.doc_ids.index(doc_id)
                doc = self.documents[idx]
                embedding = self._get_embedding(doc['text'])
                scores, _ = self.index.search(embedding, 1)
                return float(scores[0][0])
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting document score: {str(e)}")
            return 0.0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Dictionary of stats
        """
        return {
            "document_count": len(self.documents),
            "index_size_mb": self.index.ntotal * self.dimension * 4 / (1024 * 1024),  # Approximate size in MB
            "last_updated": None  # Could add timestamp if needed
        }
            
    def close(self):
        """Clean up resources."""
        try:
            # Clear FAISS index
            if hasattr(self, 'index'):
                self.index.reset()
            # Clear document storage
            self.documents = []
            self.doc_ids = []
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
