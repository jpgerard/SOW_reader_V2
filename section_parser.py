"""
Section Parser Module

Specialized module for parsing document sections with robust section number extraction.
Handles various section numbering formats and maintains hierarchical structure.
"""

import re
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a document section with hierarchical information."""
    id: str
    title: str
    level: int
    content: List[str]
    parent_id: Optional[str] = None
    subsections: List['Section'] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

class SectionParser:
    """Handles section parsing and hierarchical structure analysis."""
    
    def __init__(self):
        # Enhanced section patterns with named groups and hierarchy support
        self.section_patterns = [
            # Letter sections (A, B, etc. with optional subsections)
            (r'^(?P<id>[A-Z](?:\.\d+)?)\s+(?P<title>[^.]+?)(?:\s*\.*\s*\d*\s*$|$)', 1),  # A, A.1
            
            # Numbered sections (1, 2, etc. with optional subsections)
            (r'^(?P<id>\d+(?:\.\d+)?)\s+(?P<title>[^.]+?)(?:\s*\.*\s*\d*\s*$|$)', 1),  # 1, 1.1
            
            # Section/Article keyword patterns
            (r'^(?:SECTION|Section)\s+(?P<id>\d+(?:\.\d+)?)\s*[-–:.]\s*(?P<title>[^.]+?)(?:\s*\.*\s*\d*\s*$|$)', 1),
            (r'^(?:ARTICLE|Article)\s+(?P<id>\d+(?:\.\d+)?)\s*[-–:.]\s*(?P<title>[^.]+?)(?:\s*\.*\s*\d*\s*$|$)', 1),
        ]
        
        # Patterns to clean section IDs
        self.id_cleanup_patterns = [
            (r'\.0', ''),           # Remove .0 suffixes
            (r'\.+$', ''),          # Remove trailing dots
            (r'^0+', ''),           # Remove leading zeros
            (r'\.0+', '.'),         # Remove leading zeros after dots
        ]
        
        # Patterns to identify table of contents entries
        self.toc_patterns = [
            r'^\s*TABLE OF CONTENTS\s*$',  # Exact TOC header
            r'^\s*CONTENTS\s*$',           # Exact Contents header
            r'^\s*Table of Contents\s*$',   # Exact TOC header (mixed case)
            r'^\s*Page \d+ of \d+\s*$',    # Exact page X of Y format
            r'^\s*Source-Selection-Sensitive\s*$',  # Common headers
            r'^\s*For Official Use Only\s*$',
            r'\s*\.{3,}\s*\d+\s*$',        # Dots followed by page number
            r'\s{3,}\d+\s*$',              # Multiple spaces followed by page number
        ]
        
        # Patterns for section ID normalization
        self.id_normalization_patterns = [
            (r'Section\s+(\d+)', r'\1'),  # Section 1 -> 1
            (r'SECTION\s+(\d+)', r'\1'),  # SECTION 1 -> 1
            (r'Article\s+(\d+)', r'\1'),  # Article 1 -> 1
            (r'\((\d+)\)', r'\1'),  # (1) -> 1
            (r'\(([A-Z])\)', r'\1'),  # (A) -> A
        ]

    def normalize_section_id(self, section_id: str) -> str:
        """Normalize section ID to standard format."""
        normalized = section_id
        for pattern, replacement in self.id_normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def get_section_level(self, section_id: str) -> int:
        """
        Determine section level from ID.
        
        Examples:
        A -> 1
        A.1 -> 2
        1 -> 1
        1.1 -> 2
        """
        # Remove any non-essential characters
        clean_id = self.normalize_section_id(section_id)
        
        # Count components
        if '.' in clean_id:
            return len(clean_id.split('.'))
        elif clean_id.isalpha() or clean_id.isdigit():
            return 1
        else:
            # Mixed format, count significant parts
            parts = re.findall(r'[A-Z]|\d+', clean_id)
            return len(parts)

    def extract_section_info(self, line: str) -> Optional[Tuple[str, str, int]]:
        """
        Extract section ID, title, and level from a line.
        Returns tuple of (id, title, level) or None if not a section header.
        """
        line = line.strip()
        if not line or self.is_toc_entry(line):
            return None
            
        for pattern, base_level in self.section_patterns:
            match = re.match(pattern, line)
            if match:
                section_id = match.group('id')
                title = match.group('title').strip()
                
                # Skip if title looks like a page number
                if re.match(r'^\d+$', title):
                    continue
                    
                # Clean up title
                title = re.sub(r'\s*-\s*', '-', title)  # Fix spaced hyphens
                title = re.sub(r'\s+', ' ', title)      # Normalize spaces
                
                # Calculate actual level
                level = self.get_section_level(section_id)
                if level == 1:
                    level = base_level
                
                return section_id, title, level
        return None

    def clean_section_id(self, section_id: str) -> str:
        """Clean and normalize section ID."""
        cleaned = section_id
        for pattern, replacement in self.id_cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.strip()

    def is_toc_entry(self, line: str) -> bool:
        """Check if line is a table of contents entry."""
        return any(re.match(pattern, line, re.IGNORECASE) for pattern in self.toc_patterns)

    def clean_section_content(self, content: List[str]) -> List[str]:
        """Clean section content by removing TOC entries and formatting artifacts."""
        cleaned = []
        for line in content:
            line = line.strip()
            if not line:
                continue
            
            # Skip TOC entries and formatting artifacts
            if self.is_toc_entry(line):
                continue
            
            # Skip page numbers
            if re.match(r'^\d+$', line):
                continue
            
            # Skip repeated dashes/underscores (often used for formatting)
            if re.match(r'^[-_=]{3,}$', line):
                continue
            
            # Clean up TOC-style formatting within lines
            line = re.sub(r'\s*\.{3,}\s*\d+\s*$', '', line)  # Remove trailing dots and page numbers
            line = re.sub(r'\s{3,}\d+\s*$', '', line)        # Remove trailing spaces and page numbers
            
            if line:  # Only add non-empty lines after cleaning
                cleaned.append(line)
        
        return cleaned

    def parse_sections(self, text: str) -> List[Section]:
        """Parse text into hierarchical sections with improved handling."""
        lines = text.split('\n')
        sections = []
        current_section = None
        section_stack = []
        content_buffer = []
        in_toc = False
        
        for line in lines:
            line = line.rstrip()  # Keep leading whitespace for indentation analysis
            if not line:
                continue
            
            # Check for TOC start/end
            if re.match(r'^\s*(?:TABLE OF )?CONTENTS\s*$', line, re.IGNORECASE):
                in_toc = True
                continue
            elif in_toc and not re.search(r'\.{3,}', line):  # No dots means we're out of TOC
                in_toc = False
            
            if in_toc:
                continue
            
            # Try to extract section info
            section_info = self.extract_section_info(line)
            
            if section_info:
                # Process any buffered content for the current section
                if current_section and content_buffer:
                    current_section.content.extend(self.clean_section_content(content_buffer))
                    content_buffer = []
                
                # Create new section
                section_id, title, level = section_info
                section_id = self.clean_section_id(section_id)
                
                # Update section stack based on level
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                
                # Create new section
                parent = section_stack[-1] if section_stack else None
                new_section = Section(
                    id=section_id,
                    title=title,
                    level=level,
                    content=[],
                    parent_id=parent.id if parent else None
                )
                
                # Check for duplicate sections
                existing_section = self.find_section_by_id(sections, section_id)
                if existing_section:
                    # Merge content if duplicate
                    logger.warning(f"Duplicate section found: {section_id}")
                    existing_section.content.extend(new_section.content)
                    current_section = existing_section
                    continue
                
                # Add to parent's subsections if exists
                if parent:
                    parent.subsections.append(new_section)
                else:
                    sections.append(new_section)
                
                # Update current section and stack
                current_section = new_section
                section_stack.append(current_section)
                
            else:
                # Buffer the content line
                content_buffer.append(line)
        
        # Process any remaining buffered content
        if current_section and content_buffer:
            current_section.content.extend(self.clean_section_content(content_buffer))
        
        return sections

    def find_section_by_id(self, sections: List[Section], target_id: str) -> Optional[Section]:
        """Find section by ID in section hierarchy."""
        target_id = self.normalize_section_id(target_id)
        
        def search_sections(section_list: List[Section]) -> Optional[Section]:
            for section in section_list:
                if self.normalize_section_id(section.id) == target_id:
                    return section
                if section.subsections:
                    result = search_sections(section.subsections)
                    if result:
                        return result
            return None
        
        return search_sections(sections)

    def get_section_content(self, section: Section, include_subsections: bool = True) -> str:
        """Get section content, optionally including subsection content."""
        content = '\n'.join(section.content)
        
        if include_subsections and section.subsections:
            for subsection in section.subsections:
                subcontent = self.get_section_content(subsection, include_subsections)
                if subcontent:
                    content += f"\n{subcontent}"
        
        return content

    def get_section_structure(self, sections: List[Section], indent: str = '') -> str:
        """Get human-readable section structure."""
        structure = []
        for section in sections:
            structure.append(f"{indent}{section.id}: {section.title}")
            if section.subsections:
                structure.append(
                    self.get_section_structure(section.subsections, indent + '  ')
                )
        return '\n'.join(structure)

    def validate_section_structure(self, sections: List[Section]) -> List[str]:
        """Validate section structure and return any issues found."""
        issues = []
        
        def validate_section(section: Section, parent_id: Optional[str] = None):
            # Validate section ID
            if not section.id:
                issues.append(f"Missing section ID in section with title: {section.title}")
            
            # Validate parent relationship
            if parent_id and parent_id != section.parent_id:
                issues.append(
                    f"Section {section.id} has incorrect parent ID. "
                    f"Expected {parent_id}, got {section.parent_id}"
                )
            
            # Validate level
            expected_level = self.get_section_level(section_id)
            if section.level != expected_level:
                issues.append(
                    f"Section {section.id} has incorrect level. "
                    f"Expected {expected_level}, got {section.level}"
                )
            
            # Validate subsections
            if section.subsections:
                for subsection in section.subsections:
                    validate_section(subsection, section.id)
        
        # Validate all sections
        for section in sections:
            validate_section(section)
        
        return issues
