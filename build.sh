#!/bin/bash 

# Install Python dependencies
python -m pip install -r requirements.txt

# Build Jupyter Lite
jupyter lite build --output-dir dist
