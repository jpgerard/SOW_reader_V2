"""
Module for processing Statement of Work (SOW) documents.
"""

import os
import re
from docx import Document
import PyPDF2
from section_parser import SectionParser
from nlp_extractor import NLPExtractor

class SOWProcessor:
    def __init__(self):
        """Initialize with required components."""
        self.section_parser = SectionParser()
        self.nlp_extractor = NLPExtractor()

    def _load_document(self, file_path: str) -> str:
        """Load and extract text from a document file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.docx':
            return self._extract_docx(file_path)
        elif ext == '.pdf':
            return self._extract_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        doc = Document(file_path)
        text = []
        
        for para in doc.paragraphs:
            text.append(para.text)
        
        # Clean and normalize text
        text = '\n'.join(text)
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Normalize apostrophes
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Normalize quotes
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize horizontal whitespace only
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple blank lines
        
        return text
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                # Split into lines and clean each line
                lines = page_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line:  # Only keep non-empty lines
                        line = re.sub(r'[ \t]+', ' ', line)  # Normalize horizontal whitespace
                        cleaned_lines.append(line)
                text.append('\n'.join(cleaned_lines))
        
        # Clean and normalize text
        text = '\n'.join(text)
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Normalize apostrophes
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Normalize quotes
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple blank lines
        
        return text

    def extract_requirements(self, text: str, section_id: str, section_title: str) -> list:
        """Extract requirements from document text with section context."""
        return self.nlp_extractor.extract_requirements(text, section_id, section_title)
