#!/bin/bash

echo "Starting setup..."

# Create temp directory for file uploads
echo "Creating temp directory..."
mkdir -p temp

# Install spaCy model
echo "Installing spaCy model..."
python -m pip install --upgrade pip
python -m pip install spacy
python -m spacy download en_core_web_sm || {
    echo "Failed to download spaCy model, retrying with --no-cache-dir..."
    python -m pip install --no-cache-dir spacy
    python -m spacy download en_core_web_sm
}

echo "Setup completed!"
