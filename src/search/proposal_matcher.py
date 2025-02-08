"""Proposal section matcher for SOW requirements."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import logging
import os
from anthropic import Anthropic
from .vector_search import ProposalVectorizer, SearchResult as VectorSearchResult, VectorSearchError

# Configure production logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class LLMAnalysis:
    """Analysis results from Claude."""
    is_addressed: bool
    how_addressed: str
    compliance_level: str
    confidence_rating: str
    improvement_suggestions: List[str]

@dataclass
class MatchResult:
    """Result of matching a requirement to proposal sections."""
    requirement_id: str
    requirement_text: str
    matched_sections: List[Dict[str, any]]
    confidence_score: float
    match_explanation: str
    llm_analysis: Optional[LLMAnalysis] = None
    suggested_improvements: Optional[str] = None

class MatcherError(Exception):
    """Base exception for matcher errors."""
    pass

class ProposalMatcher:
    """Matches SOW requirements to proposal sections using vector search."""
    
    def __init__(self, embedding_model=None):
        """Initialize the matcher.
        
        Args:
            embedding_model: Optional model for generating embeddings
        """
        try:
            self.vector_search = ProposalVectorizer()
            self._current_proposal = None
            self._proposal_indexed = False
            import streamlit as st
            api_key = st.secrets["general"]["ANTHROPIC_API_KEY"]
            if not api_key or api_key == 'your-api-key-here':
                raise MatcherError("ANTHROPIC_API_KEY not properly set in Streamlit secrets")
            self.anthropic = Anthropic(api_key=api_key)
            logger.info("ProposalMatcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ProposalMatcher: {str(e)}")
            raise MatcherError(f"Matcher initialization failed: {str(e)}")

    def _extract_section_id(self, text: str) -> Optional[str]:
        """Extract section ID in X.X.X format."""
        try:
            pattern = r'\b(\d+\.\d+\.\d+)\b'
            match = re.search(pattern, text)
            return match.group(1) if match else None
        except Exception as e:
            logger.error(f"Error extracting section ID: {str(e)}")
            return None

    def _calculate_section_similarity(self, req_section: str, prop_section: str) -> float:
        """Calculate similarity between section IDs."""
        if not req_section or not prop_section:
            return 0.0
            
        try:
            # Split into components
            req_parts = req_section.split('.')
            prop_parts = prop_section.split('.')
            
            # Calculate matching score based on matching prefix parts
            matching_parts = 0
            for req_part, prop_part in zip(req_parts, prop_parts):
                if req_part == prop_part:
                    matching_parts += 1
                else:
                    break
                    
            return matching_parts / len(req_parts)
        except Exception as e:
            logger.error(f"Error calculating section similarity: {str(e)}")
            return 0.0

    def _process_search_results(self, 
                              vector_results: List[VectorSearchResult],
                              requirement_section: Optional[str]) -> List[Dict[str, any]]:
        """Process vector search results."""
        processed_results = []
        
        try:
            for result in vector_results:
                section_id = self._extract_section_id(result.chunk.text)
                if section_id:
                    section_score = self._calculate_section_similarity(
                        requirement_section, section_id
                    ) if requirement_section else 0.0
                    
                    # Combine vector and section scores
                    vector_score = result.similarity_score
                    final_score = (vector_score * 0.7) + (section_score * 0.3)  # Weight vector score higher
                    
                    processed_results.append({
                        'section_id': section_id,
                        'text': result.chunk.text,
                        'vector_score': vector_score,
                        'section_score': section_score,
                        'final_score': final_score
                    })

            # Sort by final score
            return sorted(processed_results, key=lambda x: x['final_score'], reverse=True)
        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
            return []

    def _analyze_with_claude(self, requirement: str, matched_sections: List[Dict[str, any]]) -> LLMAnalysis:
        """Analyze requirement coverage using Claude."""
        # Prepare context from matched sections
        context = "\n\n".join([
            f"Section {match['section_id']}:\n{match['text']}"
            for match in matched_sections[:3]  # Use top 3 matches
        ])

        # Construct prompt
        prompt = f"""You are analyzing how well a proposal addresses a specific requirement. 

Requirement:
{requirement}

Relevant proposal sections:
{context}

Please analyze and provide:
1. Is the requirement addressed? (Yes/No)
2. How specifically is it addressed or not addressed?
3. Compliance level (Fully Compliant/Partially Compliant/Non-Compliant)
4. Confidence rating (0-100%) with explanation
5. Specific suggestions for improvement

Format your response as JSON with the following structure:
{{
    "is_addressed": boolean,
    "how_addressed": "detailed explanation",
    "compliance_level": "level",
    "confidence_rating": "percentage% - explanation",
    "improvement_suggestions": ["suggestion1", "suggestion2", ...]
}}"""

        response = None
        try:
            # Get Claude's analysis
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are an expert at analyzing how well proposal sections address specific requirements. You provide detailed, objective analysis in JSON format.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse response
            try:
                analysis = eval(response.content[0].text)  # Try eval first
            except:
                # If eval fails, try json.loads with some cleanup
                import json
                cleaned_text = response.content[0].text.strip()
                # Remove any markdown code block markers
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                analysis = json.loads(cleaned_text)
            
            # Ensure all required fields are present
            required_fields = {
                "is_addressed": False,
                "how_addressed": "Analysis failed - missing field",
                "compliance_level": "Unknown",
                "confidence_rating": "0% - Missing field",
                "improvement_suggestions": ["Analysis incomplete"]
            }
            
            # Merge with defaults for any missing fields
            analysis = {**required_fields, **analysis}
            
            return LLMAnalysis(
                is_addressed=analysis["is_addressed"],
                how_addressed=analysis["how_addressed"],
                compliance_level=analysis["compliance_level"],
                confidence_rating=analysis["confidence_rating"],
                improvement_suggestions=analysis["improvement_suggestions"]
            )

        except Exception as e:
            logger.error(f"Error in Claude analysis: {str(e)}")
            logger.error(f"Prompt used: {prompt}")
            if response and hasattr(response, 'content') and response.content:
                logger.error(f"Raw response: {response.content[0].text}")
            return LLMAnalysis(
                is_addressed=False,
                how_addressed=f"Error in analysis: {str(e)}",
                compliance_level="Error",
                confidence_rating="0% - Analysis failed",
                improvement_suggestions=["Analysis failed due to technical error"]
            )

    def _generate_match_explanation(self, 
                                  requirement: str,
                                  matches: List[Dict[str, any]]) -> Tuple[str, Optional[str]]:
        """Generate explanation and improvement suggestions for matches."""
        try:
            if not matches:
                return "No matching sections found.", "Consider adding a section that directly addresses this requirement."

            top_match = matches[0]
            
            # Generate match explanation
            explanation_parts = [
                f"Best match found in section {top_match['section_id']} ",
                f"(Confidence: {top_match['final_score']:.2f})"
            ]
            
            # Add score breakdown
            explanation_parts.append("\nScore breakdown:")
            explanation_parts.append(f"- Semantic similarity: {top_match['vector_score']:.2f}")
            explanation_parts.append(f"- Section matching: {top_match['section_score']:.2f}")
            
            explanation = "\n".join(explanation_parts)
            
            # Generate improvement suggestions
            improvements = None
            if top_match['final_score'] < 0.7:  # Threshold for suggesting improvements
                improvements = []
                if top_match['vector_score'] < 0.6:
                    improvements.append(
                        "Consider using more similar terminology to the requirement"
                    )
                if top_match['section_score'] < 0.6:
                    improvements.append(
                        "Consider reorganizing content to better align with the requirement's section structure"
                    )
                
                improvements = "\n".join(improvements) if improvements else None
                
            return explanation, improvements
        except Exception as e:
            logger.error(f"Error generating match explanation: {str(e)}")
            return "Error generating explanation.", None

    def match_requirement(self, 
                         requirement_text: str,
                         proposal_text: str,
                         requirement_id: Optional[str] = None) -> MatchResult:
        """Match a requirement to sections in the proposal.
        
        Args:
            requirement_text: The requirement text to match
            proposal_text: The proposal text to search in
            requirement_id: Optional requirement ID for better section matching
            
        Returns:
            MatchResult containing matched sections and scoring information
            
        Raises:
            MatcherError: If matching fails
        """
        try:
            # Only reindex if proposal text has changed
            if proposal_text != self._current_proposal:
                logger.info("New proposal text detected, reindexing...")
                self.vector_search.index_proposal(proposal_text)
                self._current_proposal = proposal_text
                self._proposal_indexed = True
            else:
                logger.info("Using cached proposal index")
                
            # Extract section ID from requirement if not provided
            if not requirement_id:
                requirement_id = self._extract_section_id(requirement_text)
                
            # Perform vector search
            vector_results = self.vector_search.search(requirement_text, top_k=5)
            
            # Process results
            matched_sections = self._process_search_results(
                vector_results,
                requirement_id
            )
            
            # Calculate overall confidence score
            confidence_score = matched_sections[0]['final_score'] if matched_sections else 0.0
            
            # Generate explanation and improvements
            explanation, improvements = self._generate_match_explanation(
                requirement_text,
                matched_sections
            )
            
            # Get LLM analysis
            llm_analysis = self._analyze_with_claude(requirement_text, matched_sections)
            
            logger.info(f"Matched requirement with confidence score: {confidence_score:.2f}")
            
            return MatchResult(
                requirement_id=requirement_id or "UNKNOWN",
                requirement_text=requirement_text,
                matched_sections=matched_sections,
                confidence_score=confidence_score,
                match_explanation=explanation,
                llm_analysis=llm_analysis,
                suggested_improvements=improvements
            )
            
        except VectorSearchError as e:
            logger.error(f"Vector search error: {str(e)}")
            raise MatcherError(f"Vector search failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error matching requirement: {str(e)}")
            raise MatcherError(f"Requirement matching failed: {str(e)}")
