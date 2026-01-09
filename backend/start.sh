#!/bin/bash

# Backend startup script for Interview Coaching System

echo "🚀 Starting Interview Coaching Backend..."

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📡 WebSocket endpoints:"
echo "   - Video: ws://localhost:8000/ws/video/{client_id}"
echo "   - Audio: ws://localhost:8000/ws/audio/{client_id}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

 # If a CLI named `uv` is available, prefer it (user requested `uv`).
 # Otherwise fall back to running the cross-platform `main.py`.
 if command -v uv >/dev/null 2>&1; then
     echo "Using 'uv' CLI to start the server"
     uv main:app --host 0.0.0.0 --port 8000 --reload
 else
     echo "'uv' not found — using python main.py"
     python main.py
 fi
