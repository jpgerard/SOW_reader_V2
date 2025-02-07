#!/bin/bash

# Make script executable
chmod +x setup.sh

# Install spaCy model
python -m spacy download en_core_web_sm

# Create temp directory for file uploads
mkdir -p temp
