"""Vector search and proposal matching functionality."""

from .vector_search import ProposalVectorizer, VectorSearchError, SearchResult, ChunkMetadata
from .proposal_matcher import ProposalMatcher, MatcherError, MatchResult, LLMAnalysis

__all__ = [
    'ProposalVectorizer',
    'VectorSearchError',
    'SearchResult',
    'ChunkMetadata',
    'ProposalMatcher',
    'MatcherError',
    'MatchResult',
    'LLMAnalysis'
]
