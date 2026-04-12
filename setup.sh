#!/usr/bin/env bash
# setup.sh — One-shot setup for Local Second Brain
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧠  Local Second Brain — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Python virtual environment
echo ""
echo "▶ Creating virtual environment…"
python3 -m venv venv
source venv/bin/activate

# 2. Install Python deps
echo "▶ Installing Python dependencies…"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Python deps installed"

# 3. Ollama check
echo ""
echo "▶ Checking Ollama…"
if ! command -v ollama &> /dev/null; then
    echo "  ✗ Ollama not found."
    echo "  Install from https://ollama.com/download"
    echo "  Then run: ollama pull gemma2:2b"
    echo "  (or gemma2:9b for better quality)"
else
    echo "  ✓ Ollama found"
    echo "▶ Pulling gemma2:2b model (this may take a few minutes)…"
    ollama pull gemma2:2b || echo "  [warning] Could not pull model — make sure ollama is running"
fi

# 4. Sample data
echo ""
echo "▶ Creating sample data…"
mkdir -p data

cat > data/sample_note.md << 'EOF'
# Notes on EEG Signal Processing

Explored bandpass filtering for alpha waves (8-12Hz) using scipy.signal.
The main challenge is artifact removal — eye blinks cause huge amplitude spikes.

## Ideas
- Could combine EEG + OpenCV for a brain-computer interface that responds to gaze
- Might apply this to focus tracking during coding sessions

## Open questions
- What's the minimum electrode count for reliable focus detection?
- Is consumer-grade (Muse headset) data good enough for serious work?
EOF

cat > data/sample_project.py << 'EOF'
"""
OpenCV face detection + emotion classification prototype.
Uses Haar cascades for detection, FER+ model for emotion.
"""
import cv2

def detect_faces(frame):
    """Returns list of (x, y, w, h) bounding boxes."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, 1.1, 4)

# TODO: plug in emotion classifier here
# Idea: combine with EEG focus data for multi-modal attention tracking
EOF

echo "  ✓ Sample data created in /data"

# 5. Done
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Add your notes/PDFs/code to the /data folder"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python app.py --reindex"
echo "  4. Run: python app.py"
echo ""
echo "  For the web UI:"
echo "  streamlit run streamlit_app.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
