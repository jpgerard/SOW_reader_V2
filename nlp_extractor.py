"""
NLP-based extractor for SOW requirements and entities.
Uses Named Entity Recognition (NER) and dependency parsing to identify key information.
"""

import re
import spacy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Requirement:
    """Represents an extracted requirement with context."""
    text: str
    type: str  # e.g., 'deliverable', 'timeline', 'technical', etc.
    section_id: str
    section_title: str
    confidence: float
    entities: Dict[str, List[str]]  # e.g., {'DATE': ['30 days'], 'ORG': ['NLRB']}

class NLPExtractor:
    """Extracts requirements and entities using NLP techniques."""
    
    def __init__(self):
        # Initialize spaCy model with fallback
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            import streamlit as st
            st.warning("Using basic NLP pipeline. Some features may be limited.")
            # Create minimal pipeline for basic functionality
            self.nlp = spacy.blank("en")
            # Add essential components
            self.nlp.add_pipe("sentencizer")
        
        # Common requirement indicators
        self.requirement_patterns = [
            # Core patterns that work well across documents
            r"(?i)shall\s+",
            r"(?i)must\s+",
            r"(?i)required\s+to\s+",
            r"(?i)responsible\s+for\s+",
            r"(?i)will\s+provide\s+",
            r"(?i)will\s+be\s+responsible\s+for\s+",
            r"(?i)will\s+ensure\s+",
            r"(?i)shall\s+ensure\s+",
            r"(?i)must\s+ensure\s+",
            # Timeline patterns
            r"(?i)within\s+\d+\s+(?:day|week|month|year)",
            r"(?i)no\s+later\s+than",
            r"(?i)by\s+the\s+end\s+of",
        ]
        
        # Entity type mappings with weighted categories
        self.entity_types = {
            'DATE': ['timeline', 'deadline', 'duration'],
            'ORG': ['organization'],  # Reduced to avoid over-categorizing
            'MONEY': ['cost', 'price', 'budget'],
            'QUANTITY': ['quantity', 'amount', 'number'],
            'PRODUCT': ['deliverable', 'artifact', 'document'],
        }
        
        # Requirement type indicators (weighted keywords)
        self.type_indicators = {
            'technical': [
                ('system', 2), ('software', 2), ('hardware', 2), ('technical', 2),
                ('configuration', 1), ('install', 1), ('implement', 1), ('deploy', 1)
            ],
            'security': [
                ('security', 2), ('protect', 2), ('encrypt', 2), ('safeguard', 2),
                ('confidential', 1), ('sensitive', 1), ('authentication', 1)
            ],
            'deliverable': [
                ('deliver', 2), ('provide', 2), ('submit', 2), ('documentation', 2),
                ('report', 1), ('plan', 1), ('document', 1)
            ],
            'timeline': [
                ('schedule', 2), ('timeline', 2), ('date', 2), ('period', 2),
                ('days', 1), ('months', 1), ('years', 1), ('duration', 1)
            ],
        }
    
    def extract_requirements(self, section_text: str, section_id: str, section_title: str) -> List[Requirement]:
        """Extract requirements from section text using NLP techniques."""
        requirements = []
        doc = self.nlp(section_text)
        
        # Process each sentence
        for sent in doc.sents:
            # Skip short sentences and section titles
            if len(sent.text.split()) < 4 or sent.text.strip() == section_title:
                continue
                
            # Check if sentence contains requirement indicators
            if any(re.search(pattern, sent.text) for pattern in self.requirement_patterns):
                # Extract entities
                entities = self._extract_entities(sent)
                
                # Determine requirement type based on entities and context
                req_type = self._determine_requirement_type(sent, entities)
                
                # Calculate confidence based on indicators and entities
                confidence = self._calculate_confidence(sent, entities)
                
                # Clean requirement text
                text = self._clean_requirement_text(sent.text)
                
                # Create requirement
                req = Requirement(
                    text=text,
                    type=req_type,
                    section_id=section_id,
                    section_title=section_title,
                    confidence=confidence,
                    entities=entities
                )
                requirements.append(req)
        
        return requirements
    
    def _extract_entities(self, sent) -> Dict[str, List[str]]:
        """Extract and categorize named entities from sentence."""
        entities = {}
        for ent in sent.ents:
            if ent.label_ in self.entity_types:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                # Clean and normalize entity text
                entity_text = re.sub(r'\s+', ' ', ent.text).strip()
                if entity_text not in entities[ent.label_]:
                    entities[ent.label_].append(entity_text)
        return entities
    
    def _determine_requirement_type(self, sent, entities: Dict[str, List[str]]) -> str:
        """Determine requirement type based on entities and context with weighted scoring."""
        text = sent.text.lower()
        scores = {req_type: 0 for req_type in self.type_indicators}
        
        # Score based on type indicators
        for req_type, indicators in self.type_indicators.items():
            for word, weight in indicators:
                if word in text:
                    scores[req_type] += weight
        
        # Add entity-based scores
        for ent_type, categories in self.entity_types.items():
            if ent_type in entities:
                for category in categories:
                    if category in scores:
                        scores[category] += 2  # Higher weight for entity matches
        
        # Get type with highest score
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return 'general'  # Default type if no strong indicators
    
    def _calculate_confidence(self, sent, entities: Dict[str, List[str]]) -> float:
        """Calculate confidence score for requirement extraction."""
        score = 0.0
        
        # Strong requirement indicators
        if any(re.search(pattern, sent.text) for pattern in self.requirement_patterns):
            score += 0.4
        
        # Relevant entities
        if entities:
            score += min(len(entities) * 0.1, 0.3)  # Cap at 0.3
        
        # Sentence structure (look for subject-verb-object)
        has_subj = has_verb = has_obj = False
        for token in sent:
            if token.dep_ == 'nsubj':
                has_subj = True
            elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                has_verb = True
            elif token.dep_ in ['dobj', 'pobj']:
                has_obj = True
        
        if has_subj and has_verb and has_obj:
            score += 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _clean_requirement_text(self, text: str) -> str:
        """Clean and normalize requirement text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common formatting artifacts
        text = re.sub(r'\s*\.\.\.*\s*\d+\s*$', '', text)  # Remove trailing dots and page numbers
        text = re.sub(r'\s*-\s*', '-', text)  # Fix spaced hyphens
        
        # Remove section references at start
        text = re.sub(r'^(?:Section|SECTION)\s+[-\w.]+\s*[:.-]\s*', '', text)
        
        return text
