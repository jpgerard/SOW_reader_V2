# SOW Analyzer

A Streamlit web application for analyzing Statement of Work (SOW) documents and matching them against RFP responses.

## Features

- PDF and DOCX document support
- Advanced section parsing
- NLP-based requirement extraction
- Entity recognition
- RFP compliance analysis
- Export to CSV/Excel

## Requirements

- Python 3.8+
- spaCy with en_core_web_sm model
- See requirements.txt for full dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Set environment variables:
- ANTHROPIC_API_KEY: Your Claude API key for RFP matching

3. Run locally:
```bash
streamlit run app.py
```

## Deployment

1. Required files:
- Core: app.py, sow_processor.py, section_parser.py, nlp_extractor.py
- Config: requirements.txt, packages.txt, setup.sh, .streamlit/config.toml
- Documentation: README.md

2. Streamlit Cloud deployment:
- Connect GitHub repository
- Set environment variables in Streamlit Cloud settings
- Deploy

## Usage

1. Upload SOW document (PDF/DOCX)
2. Optional: Upload proposal document for matching
3. View extracted requirements and analysis
4. Export results to CSV/Excel

## Architecture

- Document Processing: PDF/DOCX text extraction
- Section Parser: Hierarchical document structure analysis
- NLP Extractor: Requirement identification and classification
- RFP Matcher: Claude-powered proposal compliance analysis

## License

MIT License
