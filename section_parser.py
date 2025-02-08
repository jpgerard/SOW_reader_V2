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
        # Generic section patterns that match common formats
        self.section_patterns = [
            # Match any letter/number followed by title
            (r'^\s*(?P<id>[A-Z0-9](?:[-.]\d+)*)\s+(?P<title>[^.]+?)(?:\s*\.+\s*\d*\s*$|$)', 1),
            
            # Match section/article keywords
            (r'^\s*(?:Section|SECTION|Article|ARTICLE)\s*(?P<id>\d+(?:[-.]\d+)*)\s*[-.:)]\s*(?P<title>.+?)(?:\s*\.+\s*\d*\s*$|$)', 1),
            
            # Match parenthesized numbers/letters
            (r'^\s*\((?P<id>[A-Z0-9](?:[-.]\d+)*)\)\s+(?P<title>[^.]+?)(?:\s*\.+\s*\d*\s*$|$)', 1),
        ]
        
        # Patterns to clean section IDs
        self.id_cleanup_patterns = [
            (r'\.0', ''),           # Remove .0 suffixes
            (r'\.+$', ''),          # Remove trailing dots
            (r'^0+', ''),           # Remove leading zeros
            (r'\.0+', '.'),         # Remove leading zeros after dots
            (r'-', '.'),            # Convert dashes to dots
        ]
        
        # Patterns to identify table of contents entries
        self.toc_patterns = [
            r'^\s*TABLE OF CONTENTS\s*$',
            r'^\s*CONTENTS\s*$',
            r'^\s*Table of Contents\s*$',
            r'^\s*Page \d+ of \d+\s*$',
            r'^\s*Source-Selection-Sensitive\s*$',
            r'^\s*For Official Use Only\s*$',
        ]
        
        # Patterns for section ID normalization
        self.id_normalization_patterns = [
            (r'Section\s+(\d+)', r'\1'),
            (r'SECTION\s+(\d+)', r'\1'),
            (r'Article\s+(\d+)', r'\1'),
            (r'\((\d+)\)', r'\1'),
            (r'\(([A-Z])\)', r'\1'),
            (r'-', '.'),            # Convert dashes to dots
        ]

    def normalize_section_id(self, section_id: str) -> str:
        """Normalize section ID to standard format."""
        normalized = section_id
        for pattern, replacement in self.id_normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def get_section_level(self, section_id: str) -> int:
        """Determine section level from ID."""
        clean_id = self.normalize_section_id(section_id)
        
        # Split on dots or dashes
        parts = re.split(r'[.-]', clean_id)
        if len(parts) > 1:
            return len(parts)
            
        # Single letter or number
        if len(clean_id) == 1:
            return 1
            
        # Count parts separated by any delimiter
        parts = re.findall(r'[A-Z]|\d+', clean_id)
        return len(parts)

    def extract_section_info(self, line: str) -> Optional[Tuple[str, str, int]]:
        """Extract section ID, title, and level from a line."""
        line = line.strip()
        if not line:
            return None
            
        # Skip TOC entries and headers/footers
        if self.is_toc_entry(line):
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
                title = re.sub(r'\s*-\s*', '-', title)
                title = re.sub(r'\s+', ' ', title)
                # Remove trailing dots and page numbers
                title = re.sub(r'\s*\.{2,}\s*\d*\s*$', '', title)
                
                # Calculate level
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
        """Clean section content."""
        cleaned = []
        for line in content:
            line = line.strip()
            if not line:
                continue
            
            # Skip TOC entries and formatting artifacts
            if self.is_toc_entry(line):
                continue
            
            # Skip page numbers and formatting lines
            if re.match(r'^\d+$', line) or re.match(r'^[-_=]{3,}$', line):
                continue
            
            # Clean up formatting
            line = re.sub(r'\s*\.{2,}\s*\d*\s*$', '', line)
            line = re.sub(r'\s{3,}\d+\s*$', '', line)
            
            if line:
                cleaned.append(line)
        
        return cleaned

    def is_valid_section(self, section: Section, parent: Optional[Section] = None) -> bool:
        """Determine if a section is valid based on context."""
        # Check if section ID follows parent's pattern
        if parent:
            parent_id = self.normalize_section_id(parent.id)
            section_id = self.normalize_section_id(section.id)
            
            # Check if section ID starts with parent ID
            if section_id.startswith(parent_id + '.'):
                return True
                
            # Allow mixed formats (e.g., "A" parent with "A-1" child)
            if section_id.startswith(parent_id) and section_id[len(parent_id)] in '.-':
                return True
                
            return False
        
        # For top-level sections
        section_id = self.normalize_section_id(section.id)
        
        # Must be a simple identifier (single letter/number or with simple separators)
        if not re.match(r'^[A-Z0-9](?:[.-]\d+)*$', section_id):
            return False
            
        return True

    def parse_sections(self, text: str) -> List[Section]:
        """Parse text into hierarchical sections."""
        lines = text.split('\n')
        sections = []
        current_section = None
        section_stack = []
        content_buffer = []
        in_toc = False
        seen_sections = set()  # Track seen section IDs
        
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            
            # Check for TOC start/end
            if re.match(r'^\s*(?:TABLE OF )?CONTENTS\s*$', line, re.IGNORECASE):
                in_toc = True
                continue
            elif in_toc and not self.is_toc_entry(line):
                in_toc = False
            
            if in_toc:
                continue
            
            # Try to extract section info
            section_info = self.extract_section_info(line)
            
            if section_info:
                # Process buffered content
                if current_section and content_buffer:
                    current_section.content.extend(self.clean_section_content(content_buffer))
                    content_buffer = []
                
                # Create new section
                section_id, title, level = section_info
                section_id = self.clean_section_id(section_id)
                
                # Skip if we've seen this section before
                if section_id in seen_sections:
                    continue
                
                # Update section stack
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                
                # Create section
                parent = section_stack[-1] if section_stack else None
                new_section = Section(
                    id=section_id,
                    title=title,
                    level=level,
                    content=[],
                    parent_id=parent.id if parent else None
                )
                
                # Only add if it's a valid section
                if self.is_valid_section(new_section, parent):
                    seen_sections.add(section_id)
                    
                    # Add to hierarchy
                    if parent:
                        parent.subsections.append(new_section)
                    else:
                        sections.append(new_section)
                    
                    current_section = new_section
                    section_stack.append(current_section)
                
            else:
                content_buffer.append(line)
        
        # Process remaining content
        if current_section and content_buffer:
            current_section.content.extend(self.clean_section_content(content_buffer))
        
        return sections

    def find_section_by_id(self, sections: List[Section], target_id: str) -> Optional[Section]:
        """Find section by ID in hierarchy."""
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
        """Get section content."""
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
        """Validate section structure."""
        issues = []
        
        def validate_section(section: Section, parent_id: Optional[str] = None):
            if not section.id:
                issues.append(f"Missing section ID in section with title: {section.title}")
            
            if parent_id and parent_id != section.parent_id:
                issues.append(
                    f"Section {section.id} has incorrect parent ID. "
                    f"Expected {parent_id}, got {section.parent_id}"
                )
            
            expected_level = self.get_section_level(section_id)
            if section.level != expected_level:
                issues.append(
                    f"Section {section.id} has incorrect level. "
                    f"Expected {expected_level}, got {section.level}"
                )
            
            if section.subsections:
                for subsection in section.subsections:
                    validate_section(subsection, section.id)
        
        for section in sections:
            validate_section(section)
        
        return issues
