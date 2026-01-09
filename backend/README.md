# Interview Coaching Backend

Real-time interview coaching backend providing:
- **Head Pose Detection**: Real-time webcam analysis for eye contact and attention tracking
- **Speech Analysis**: Speech-to-text using OpenAI Whisper with clarity and fluency scoring

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Or use the startup script:
```bash
chmod +x start.sh
./start.sh
```

### 3. API Endpoints

#### REST Endpoints

- `GET /` - Health check and API info
- `GET /health` - Detailed health check
- `POST /api/analyze-audio` - Analyze uploaded audio file
- `POST /api/analyze-frame` - Analyze single video frame

#### WebSocket Endpoints

- `ws://localhost:8000/ws/video/{client_id}` - Real-time video frame processing
- `ws://localhost:8000/ws/audio/{client_id}` - Real-time audio processing
- `ws://localhost:8000/ws/combined/{client_id}` - Combined video + audio processing

## WebSocket Protocol

### Video WebSocket

Send:
```json
{
  "frame": "data:image/jpeg;base64,..."
}
```

Receive:
```json
{
  "yaw": 5.2,
  "pitch": -3.1,
  "roll": 1.0,
  "is_looking_at_camera": true,
  "attention_score": 92.5,
  "eye_contact_score": 88.0,
  "face_detected": true,
  "timestamp": 1702847200.123
}
```

### Audio WebSocket

Send:
```json
{
  "audio": "base64-encoded-float32-samples",
  "sample_rate": 16000
}
```

Or command:
```json
{
  "command": "analyze"
}
```

Receive:
```json
{
  "transcribed_text": "I have experience in...",
  "clarity_score": 85.5,
  "fluency_score": 78.2,
  "pace_wpm": 145.3,
  "filler_words_count": 2,
  "confidence": 0.92,
  "pronunciation_score": 88.0,
  "timestamp": 1702847200.123
}
```

## Requirements

- Python 3.9+
- Webcam (for video processing)
- Microphone (for audio processing)
- NVIDIA GPU (optional, for faster Whisper processing)

## Model Sizes

The speech analyzer uses OpenAI Whisper. Available model sizes:
- `tiny` - Fastest, least accurate (~39M parameters)
- `base` - Good balance (default, ~74M parameters)
- `small` - Better accuracy (~244M parameters)
- `medium` - High accuracy (~769M parameters)
- `large` - Best accuracy (~1550M parameters)

Change the model size in `server.py`:
```python
speech_analyzer = get_speech_analyzer(model_size="small")
```
