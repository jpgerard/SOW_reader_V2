"""
SOW Analyzer Web Application
"""

from typing import Dict
import re
import streamlit as st
from pathlib import Path
import pandas as pd
import os
import json
import io
from dotenv import load_dotenv
import anthropic
import pdfplumber
from docx import Document
import logging
from sow_processor import SOWProcessor
from src.search.proposal_matcher import ProposalMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if ANTHROPIC_API_KEY is set and valid"""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = st.secrets["general"]["ANTHROPIC_API_KEY"]
        if not api_key or api_key == 'your-api-key-here':
            st.error('Please set your ANTHROPIC_API_KEY in environment variables or Streamlit secrets')
            return False, None
        return True, api_key
    except Exception:
        st.error('ANTHROPIC_API_KEY not found in environment variables or Streamlit secrets')
        return False, None

def extract_proposal_sections(proposal_text: str) -> Dict[str, str]:
    """Extract sections from proposal text."""
    sections = {}
    # Extract sections with format X.X.X
    matches = re.finditer(r'(?:(?<=\n)|^)(\d+\.\d+\.\d+)\s*(.*?)(?=\n\s*\d+\.\d+\.\d+|$)', proposal_text, re.DOTALL)
    for match in matches:
        section_num = match.group(1)
        content = match.group(2).strip()
        sections[section_num] = content
    return sections

def init_temp_dir() -> Path:
    """Initialize temp directory."""
    if 'temp_dir' not in st.session_state:
        # Create temp directory in current working directory
        temp_dir = Path(os.getcwd()) / 'temp'
        temp_dir.mkdir(exist_ok=True, parents=True)
        st.session_state.temp_dir = temp_dir
    return st.session_state.temp_dir

def cleanup_temp_files():
    """Clean up any temporary files."""
    if 'temp_dir' in st.session_state:
        temp_dir = st.session_state.temp_dir
        if temp_dir.exists():
            for temp_file in temp_dir.glob('*'):
                try:
                    temp_file.unlink()
                except Exception as e:
                    st.error(f"Error deleting temp file {temp_file}: {str(e)}")

st.set_page_config(
    page_title="SOW Analyzer",
    page_icon="📄",
    layout="wide"
)

[Rest of the file content remains the same...]
