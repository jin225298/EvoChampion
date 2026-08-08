#!/bin/bash

echo "================================"
echo "EvoChampion Demo - Quick Start"
echo "================================"
echo ""

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file. Please edit it with your configuration."
    echo ""
fi

echo "Installing dependencies..."
pip3 install -q -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Usage:"
echo "  python3 main.py 'your goal here'"
echo ""
echo "Example:"
echo "  python3 main.py '提高数学能力'"
echo ""
