#!/bin/bash

echo "Starting setup..."

# Create temp directory for file uploads
echo "Creating temp directory..."
mkdir -p temp

# Install spaCy and model
echo "Installing spaCy and model..."
python -m pip install --upgrade pip
python -m pip install --no-cache-dir spacy
python -m pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz

# Verify installation
echo "Verifying spaCy installation..."
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('SpaCy model loaded successfully!')" || {
    echo "Failed to verify spaCy installation. Retrying direct download..."
    python -m spacy download en_core_web_sm
}

echo "Setup completed!"
