"""Enhanced matcher implementation optimized for Streamlit Cloud."""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Result from matching a requirement against proposal sections."""
    requirement_id: str
    matched_sections: List[Dict[str, Any]]
    suggested_improvements: Optional[List[str]] = None

class EnhancedMatcher:
    """Enhanced matcher with optimizations for Streamlit Cloud."""
    
    def __init__(self, embedding_model: Optional[str] = "all-mpnet-base-v2"):
        """Initialize the matcher.
        
        Args:
            embedding_model: Optional pre-trained model for embeddings
        """
        try:
            logger.info("Initializing enhanced matcher")
            # Force CPU usage for Streamlit Cloud compatibility
            self.device = torch.device("cpu")
            
            if embedding_model:
                self.model = SentenceTransformer(embedding_model, device=self.device)
            else:
                self.model = None
                
            self.sections = {}
            self.section_embeddings = {}
            self.communities = defaultdict(list)
            
        except Exception as e:
            logger.error(f"Failed to initialize matcher: {str(e)}")
            raise
            
    def match(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching sections for a query.
        
        Args:
            query: Search query
            top_k: Number of top matches to return
            
        Returns:
            List of matching sections with scores
        """
        try:
            if not self.sections:
                return []
                
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Calculate similarities
            matches = []
            for section_id, section in self.sections.items():
                if section_id in self.section_embeddings:
                    similarity = np.dot(query_embedding, self.section_embeddings[section_id])
                    matches.append({
                        'id': section_id,
                        'content': section['text'],
                        'score': float(similarity),
                        'metadata': section.get('metadata', {})
                    })
            
            # Sort by score
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error performing match: {str(e)}")
            raise
            
    def match_requirement(self, requirement_text: str, proposal_text: str, requirement_id: str) -> MatchResult:
        """Match a requirement against proposal sections.
        
        Args:
            requirement_text: The requirement text
            proposal_text: The proposal text
            requirement_id: Identifier for the requirement
            
        Returns:
            MatchResult with matched sections and suggestions
        """
        try:
            # Split proposal into sections
            sections = self._split_into_sections(proposal_text)
            
            # Generate embeddings
            requirement_embedding = self.model.encode(requirement_text, convert_to_numpy=True)
            section_embeddings = {}
            
            for section_id, section in sections.items():
                section_embedding = self.model.encode(section['text'], convert_to_numpy=True)
                section_embeddings[section_id] = section_embedding
                
            # Find matches
            matches = []
            for section_id, section in sections.items():
                similarity = np.dot(requirement_embedding, section_embeddings[section_id])
                if similarity > 0.5:  # Minimum similarity threshold
                    matches.append({
                        'section_id': section_id,
                        'text': section['text'],
                        'final_score': float(similarity)
                    })
            
            # Sort matches by score
            matches.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Generate suggestions if needed
            suggestions = []
            if not matches or matches[0]['final_score'] < 0.7:
                suggestions.append("Consider adding more specific details addressing this requirement")
                
            return MatchResult(
                requirement_id=requirement_id,
                matched_sections=matches,
                suggested_improvements=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error matching requirement: {str(e)}")
            raise
            
    def _split_into_sections(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Split text into logical sections.
        
        Args:
            text: Text to split
            
        Returns:
            Dictionary of section IDs to section content
        """
        # Simple section splitting by paragraphs
        sections = {}
        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                section_id = f"s{i+1}"
                sections[section_id] = {
                    'text': para.strip(),
                    'metadata': {'position': i}
                }
                
        return sections
        
    def get_score(self, doc_id: str) -> float:
        """Get the score for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Score between 0 and 1
        """
        try:
            if doc_id in self.sections and doc_id in self.section_embeddings:
                embedding = self.section_embeddings[doc_id]
                return float(np.linalg.norm(embedding))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting document score: {str(e)}")
            return 0.0
            
    def get_community(self, doc_id: str) -> Optional[int]:
        """Get community ID for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Community ID if available
        """
        for community_id, members in self.communities.items():
            if doc_id in members:
                return community_id
        return None
        
    def get_communities(self) -> Dict[str, int]:
        """Get all document community assignments.
        
        Returns:
            Dictionary mapping document IDs to community IDs
        """
        communities = {}
        for community_id, members in self.communities.items():
            for doc_id in members:
                communities[doc_id] = community_id
        return communities
        
    def close(self):
        """Clean up resources."""
        try:
            self.sections.clear()
            self.section_embeddings.clear()
            self.communities.clear()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
