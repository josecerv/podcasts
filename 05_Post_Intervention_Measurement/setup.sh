#!/bin/bash
# Setup script for podcast guest analysis

echo "Podcast Guest Analysis - Setup"
echo "=============================="
echo ""

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment active"
    echo "   Recommended: python3 -m venv venv && source venv/bin/activate"
fi

echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the analysis:"
echo "  python analyze.py"