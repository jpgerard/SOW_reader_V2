"""Hybrid search combining enhanced vector search with matcher, optimized for Streamlit Cloud."""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging
from .enhanced_vector_search import EnhancedVectorizer, SearchResult
from .enhanced_matcher import EnhancedMatcher

logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    vector_weight: float = 0.6
    entity_weight: float = 0.4
    use_community_boost: bool = True

class HybridSearchEngine:
    """Enhanced search engine optimized for Streamlit Cloud."""
    
    def __init__(self, embedding_model: str = "all-mpnet-base-v2", **kwargs):
        """Initialize the hybrid search engine.
        
        Args:
            embedding_model: Model name for vector search
            **kwargs: Additional configuration options
        """
        try:
            logger.info("Initializing hybrid search engine")
            self.vector_search = EnhancedVectorizer(model_name=embedding_model)
            self.matcher = EnhancedMatcher(embedding_model=embedding_model)
            self.config = SearchConfig(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {str(e)}")
            raise
            
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform hybrid search combining vector and matcher approaches.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Get vector search results
            vector_results = self.vector_search.search(query, top_k=top_k)
            
            # Get matcher results
            matcher_results = self.matcher.match(query, top_k=top_k)
            
            # Combine results with weighted scoring
            combined_results = {}
            
            # Process vector results
            for result in vector_results:
                combined_results[result.id] = {
                    'score': result.score * self.config.vector_weight,
                    'content': result.content,
                    'metadata': result.metadata
                }
                
            # Process matcher results
            for result in matcher_results:
                if result['id'] in combined_results:
                    combined_results[result['id']]['score'] += (
                        result['score'] * self.config.entity_weight
                    )
                else:
                    combined_results[result['id']] = {
                        'score': result['score'] * self.config.entity_weight,
                        'content': result['content'],
                        'metadata': result['metadata']
                    }
            
            # Apply community boost if enabled
            if self.config.use_community_boost:
                self._apply_community_boost(combined_results)
            
            # Convert to SearchResult objects and sort
            results = []
            for id_, data in combined_results.items():
                results.append(SearchResult(
                    id=id_,
                    content=data['content'],
                    score=data['score'],
                    metadata=data['metadata']
                ))
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise
            
    def _apply_community_boost(self, results: Dict[str, Dict[str, Any]]):
        """Apply community detection boost to results.
        
        Args:
            results: Dictionary of search results to boost
        """
        try:
            communities = self.matcher.get_communities()
            
            # Group results by community
            community_groups = {}
            for id_, data in results.items():
                community = communities.get(id_)
                if community:
                    if community not in community_groups:
                        community_groups[community] = []
                    community_groups[community].append(id_)
            
            # Apply boost to results in same community
            boost_factor = 1.2
            for community, members in community_groups.items():
                if len(members) > 1:
                    for id_ in members:
                        results[id_]['score'] *= boost_factor
                        
        except Exception as e:
            logger.warning(f"Failed to apply community boost: {str(e)}")
            
    def explain_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate explanation for search results.
        
        Args:
            results: List of search results to explain
            
        Returns:
            Dictionary containing result explanations
        """
        explanations = {
            'vector_weight': self.config.vector_weight,
            'entity_weight': self.config.entity_weight,
            'community_boost': self.config.use_community_boost,
            'results': []
        }
        
        for result in results:
            explanation = {
                'id': result.id,
                'final_score': result.score,
                'vector_score': self.vector_search.get_score(result.id),
                'matcher_score': self.matcher.get_score(result.id),
                'community': self.matcher.get_community(result.id)
            }
            explanations['results'].append(explanation)
            
        return explanations
        
    def close(self):
        """Clean up resources."""
        try:
            self.vector_search.close()
            self.matcher.close()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
