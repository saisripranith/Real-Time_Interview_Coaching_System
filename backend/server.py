"""
FastAPI Backend Server for Real-Time Interview Coaching System

Provides:
1. WebSocket endpoint for real-time video frame processing (head pose detection)
2. WebSocket endpoint for real-time audio processing (speech-to-text and analysis)
3. REST endpoints for batch analysis
"""

import asyncio
import base64
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import numpy as np
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import our modules
from head_pose_detector import HeadPoseDetector, get_head_pose_detector
from speech_analyzer import SpeechAnalyzer, get_speech_analyzer, SpeechAnalysisResult


# Global instances (initialized at startup)
head_pose_detector: Optional[HeadPoseDetector] = None
speech_analyzer: Optional[SpeechAnalyzer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global head_pose_detector, speech_analyzer
    
    print("🚀 Starting Interview Coaching Backend...")
    
    # Initialize head pose detector
    print("📷 Initializing Head Pose Detector...")
    head_pose_detector = get_head_pose_detector()
    print("✅ Head Pose Detector ready!")
    
    # Initialize speech analyzer (this loads Whisper model)
    print("🎤 Initializing Speech Analyzer (loading Whisper model)...")
    # Use 'base' model for balance of speed and accuracy
    # Use 'tiny' for faster but less accurate, 'small'/'medium' for more accuracy
    speech_analyzer = get_speech_analyzer(model_size="base")
    print("✅ Speech Analyzer ready!")
    
    print("🎉 Backend fully initialized and ready!")
    
    yield
    
    # Cleanup
    print("🔄 Shutting down...")
    if head_pose_detector:
        head_pose_detector.cleanup()
    print("👋 Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="Interview Coaching API",
    description="Real-time interview coaching with head pose detection and speech analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== REST Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Interview Coaching API",
        "version": "1.0.0",
        "endpoints": {
            "websocket_video": "/ws/video",
            "websocket_audio": "/ws/audio",
            "analyze_audio": "/api/analyze-audio"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "head_pose_detector": head_pose_detector is not None,
            "speech_analyzer": speech_analyzer is not None
        }
    }


@app.post("/api/analyze-audio")
async def analyze_audio_file(
    audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None)
):
    """
    Analyze an uploaded audio file for speech metrics.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        reference_text: Optional reference text for comparison
        
    Returns:
        Speech analysis results
    """
    if speech_analyzer is None:
        raise HTTPException(status_code=503, detail="Speech analyzer not initialized")
    
    # Save uploaded file temporarily
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Analyze the audio file
        result = speech_analyzer.analyze_speech_file(tmp_path, reference_text)
        return result.to_dict()
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/api/analyze-frame")
async def analyze_single_frame(frame_data: dict):
    """
    Analyze a single video frame for head pose.
    
    Args:
        frame_data: JSON with "frame" key containing base64-encoded image
        
    Returns:
        Head pose analysis results
    """
    if head_pose_detector is None:
        raise HTTPException(status_code=503, detail="Head pose detector not initialized")
    
    frame_base64 = frame_data.get("frame")
    if not frame_base64:
        raise HTTPException(status_code=400, detail="No frame data provided")
    
    result = head_pose_detector.process_base64_frame(frame_base64)
    return result.to_dict()


# ==================== WebSocket Endpoints ====================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"📱 Client {client_id} connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        print(f"📴 Client {client_id} disconnected. Active: {len(self.active_connections)}")
    
    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)


video_manager = ConnectionManager()
audio_manager = ConnectionManager()


@app.websocket("/ws/video/{client_id}")
async def websocket_video_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time video frame processing.
    
    Client sends: JSON with "frame" key containing base64-encoded image
    Server sends: JSON with head pose analysis results and annotated frame
    """
    await video_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if "frame" not in data:
                await websocket.send_json({"error": "No frame data"})
                continue
            
            # Process frame
            if head_pose_detector is None:
                await websocket.send_json({"error": "Detector not ready"})
                continue
            
            try:
                base64_frame = data["frame"]
                
                # Decode frame for processing
                if ',' in base64_frame:
                    base64_frame = base64_frame.split(',')[1]
                
                img_bytes = base64.b64decode(base64_frame)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Could not decode frame"})
                    continue
                
                # Process frame to get head pose
                result = head_pose_detector.process_frame(frame)
                
                # Draw annotations on the frame
                annotated_frame = head_pose_detector.draw_annotations(frame, result)
                
                # Encode annotated frame back to base64
                success, buffer = cv2.imencode('.jpg', annotated_frame)
                if success:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    response = result.to_dict()
                    response["timestamp"] = time.time()
                    response["annotated_frame"] = f"data:image/jpeg;base64,{frame_base64}"
                    
                    # Debug: log detection status and angles
                    print(f"VideoWS: face_detected={response.get('face_detected')} yaw={response.get('yaw')} pitch={response.get('pitch')} roll={response.get('roll')}")
                    await websocket.send_json(response)
                else:
                    await websocket.send_json({"error": "Could not encode annotated frame"})
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        video_manager.disconnect(client_id)
    except Exception as e:
        print(f"Video WebSocket error: {e}")
        video_manager.disconnect(client_id)


@app.websocket("/ws/audio/{client_id}")
async def websocket_audio_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time audio processing.
    
    Client sends: JSON with "audio" key containing base64-encoded audio data
                  and optionally "sample_rate" (default 16000)
    Server sends: JSON with speech analysis results
    """
    await audio_manager.connect(websocket, client_id)
    
    # Buffer for accumulating audio chunks
    audio_buffer = []
    last_analysis_time = time.time()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if "audio" not in data:
                if "command" in data:
                    # Handle commands
                    if data["command"] == "analyze":
                        # Force analysis of buffered audio
                        if audio_buffer and speech_analyzer:
                            combined = np.concatenate(audio_buffer)
                            result = speech_analyzer.analyze_speech(
                                combined,
                                sample_rate=data.get("sample_rate", 16000)
                            )
                            response = result.to_dict()
                            response["timestamp"] = time.time()
                            await websocket.send_json(response)
                            audio_buffer.clear()
                    elif data["command"] == "clear":
                        audio_buffer.clear()
                continue
            
            # Decode audio data
            audio_base64 = data["audio"]
            sample_rate = data.get("sample_rate", 16000)
            
            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_base64)
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                audio_buffer.append(audio_data)
                
                # Analyze every few seconds or when buffer is large enough
                current_time = time.time()
                total_samples = sum(len(chunk) for chunk in audio_buffer)
                duration = total_samples / sample_rate
                
                # Analyze if we have at least 3 seconds of audio
                if duration >= 3.0 or (current_time - last_analysis_time >= 5.0 and duration >= 1.0):
                    if speech_analyzer:
                        combined = np.concatenate(audio_buffer)
                        result = speech_analyzer.analyze_speech(
                            combined,
                            sample_rate=sample_rate,
                            duration_seconds=duration
                        )
                        response = result.to_dict()
                        response["timestamp"] = time.time()
                        response["audio_duration"] = duration
                        await websocket.send_json(response)
                        
                        # Keep last 1 second for continuity
                        keep_samples = int(sample_rate * 1.0)
                        audio_buffer = [combined[-keep_samples:]] if len(combined) > keep_samples else []
                        last_analysis_time = current_time
                else:
                    # Send acknowledgment
                    await websocket.send_json({
                        "status": "buffering",
                        "buffered_duration": duration,
                        "timestamp": time.time()
                    })
                    
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        audio_manager.disconnect(client_id)
    except Exception as e:
        print(f"Audio WebSocket error: {e}")
        audio_manager.disconnect(client_id)


@app.websocket("/ws/combined/{client_id}")
async def websocket_combined_endpoint(websocket: WebSocket, client_id: str):
    """
    Combined WebSocket endpoint for both video and audio processing.
    
    Client sends: JSON with either "frame" or "audio" key
    Server sends: JSON with corresponding analysis results
    """
    await websocket.accept()
    print(f"📱 Combined client {client_id} connected")
    
    audio_buffer = []
    last_analysis_time = time.time()
    
    try:
        while True:
            data = await websocket.receive_json()
            response = {"timestamp": time.time()}
            
            # Handle video frame
            if "frame" in data and head_pose_detector:
                try:
                    result = head_pose_detector.process_base64_frame(data["frame"])
                    response["video"] = result.to_dict()
                except Exception as e:
                    response["video_error"] = str(e)
            
            # Handle audio data
            if "audio" in data and speech_analyzer:
                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    sample_rate = data.get("sample_rate", 16000)
                    audio_buffer.append(audio_data)
                    
                    total_samples = sum(len(chunk) for chunk in audio_buffer)
                    duration = total_samples / sample_rate
                    
                    if duration >= 3.0:
                        combined = np.concatenate(audio_buffer)
                        result = speech_analyzer.analyze_speech(
                            combined,
                            sample_rate=sample_rate,
                            duration_seconds=duration
                        )
                        response["audio"] = result.to_dict()
                        
                        keep_samples = int(sample_rate * 1.0)
                        audio_buffer = [combined[-keep_samples:]] if len(combined) > keep_samples else []
                    else:
                        response["audio_status"] = "buffering"
                        response["audio_duration"] = duration
                        
                except Exception as e:
                    response["audio_error"] = str(e)
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print(f"📴 Combined client {client_id} disconnected")
    except Exception as e:
        print(f"Combined WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
