/**
 * Custom hook for real-time interview metrics via WebSocket
 * Connects to the backend for head pose detection and speech analysis
 */

import { useState, useEffect, useRef, useCallback } from 'react';

// Backend server URL
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const WS_URL = BACKEND_URL.replace('http', 'ws');

export interface VideoMetrics {
  yaw: number;
  pitch: number;
  roll: number;
  is_looking_at_camera: boolean;
  attention_score: number;
  eye_contact_score: number;
  face_detected: boolean;
  timestamp?: number;
}

export interface AudioMetrics {
  transcribed_text: string;
  clarity_score: number;
  fluency_score: number;
  pace_wpm: number;
  filler_words_count: number;
  confidence: number;
  pronunciation_score: number;
  timestamp?: number;
}

export interface InterviewMetrics {
  video: VideoMetrics | null;
  audio: AudioMetrics | null;
  isConnected: boolean;
  isVideoActive: boolean;
  isAudioActive: boolean;
  error: string | null;
}

const DEFAULT_VIDEO_METRICS: VideoMetrics = {
  yaw: 0,
  pitch: 0,
  roll: 0,
  is_looking_at_camera: false,
  attention_score: 0,
  eye_contact_score: 0,
  face_detected: false,
};

const DEFAULT_AUDIO_METRICS: AudioMetrics = {
  transcribed_text: '',
  clarity_score: 0,
  fluency_score: 0,
  pace_wpm: 0,
  filler_words_count: 0,
  confidence: 0,
  pronunciation_score: 0,
};

export function useInterviewMetrics() {
  const [metrics, setMetrics] = useState<InterviewMetrics>({
    video: DEFAULT_VIDEO_METRICS,
    audio: DEFAULT_AUDIO_METRICS,
    isConnected: false,
    isVideoActive: false,
    isAudioActive: false,
    error: null,
  });

  const videoWsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Audio buffer for sending to backend
  const audioBufferRef = useRef<Float32Array[]>([]);
  const lastAudioSendRef = useRef<number>(0);

  // Generate a unique client ID
  const clientIdRef = useRef<string>(
    `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  );

  /**
   * Connect to the video WebSocket
   */
  const connectVideoWs = useCallback(() => {
    if (videoWsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_URL}/ws/video/${clientIdRef.current}`);

    ws.onopen = () => {
      console.log('📷 Video WebSocket connected');
      setMetrics(prev => ({ ...prev, isConnected: true, error: null }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (!data.error) {
          setMetrics(prev => ({ 
            ...prev, 
            video: { ...data, timestamp: data.timestamp || Date.now() }
          }));
        }
      } catch (e) {
        console.error('Failed to parse video response:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('Video WebSocket error:', error);
      setMetrics(prev => ({ ...prev, error: 'Video connection error' }));
    };

    ws.onclose = () => {
      console.log('📷 Video WebSocket closed');
      setMetrics(prev => ({ ...prev, isConnected: false }));
      // Try to reconnect after 2 seconds
      setTimeout(connectVideoWs, 2000);
    };

    videoWsRef.current = ws;
  }, []);

  /**
   * Send a video frame to the backend
   */
  const sendFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !videoWsRef.current) return;
    if (videoWsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx || video.videoWidth === 0) return;

    // Set canvas size to video size (use full res for reliability)
    // If performance becomes an issue, reduce to 0.75
    const scale = 1.0;
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 JPEG with slightly higher quality to aid detection
    const frameData = canvas.toDataURL('image/jpeg', 0.85);

    // Send to backend
    try {
      videoWsRef.current.send(JSON.stringify({ frame: frameData }));
    } catch (e) {
      console.error('Failed to send frame:', e);
    }
  }, []);

  /**
   * Start video capture and processing
   */
  const startVideo = useCallback(async (videoElement: HTMLVideoElement) => {
    try {
      // Get camera stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });

      videoElement.srcObject = stream;
      await videoElement.play();

      videoRef.current = videoElement;
      streamRef.current = stream;

      // Create canvas for frame capture
      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;

      // Connect WebSocket
      connectVideoWs();

      // Start sending frames at ~10 FPS
      frameIntervalRef.current = setInterval(sendFrame, 100);

      setMetrics(prev => ({ ...prev, isVideoActive: true }));
      
      return stream;
    } catch (error) {
      console.error('Failed to start video:', error);
      setMetrics(prev => ({ ...prev, error: 'Camera access denied' }));
      throw error;
    }
  }, [connectVideoWs, sendFrame]);

  /**
   * Stop video capture
   */
  const stopVideo = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoWsRef.current) {
      videoWsRef.current.close();
      videoWsRef.current = null;
    }

    setMetrics(prev => ({ ...prev, isVideoActive: false }));
  }, []);

  /**
   * Send audio data to backend for analysis
   */
  const sendAudioForAnalysis = useCallback(async (audioBlob: Blob, referenceText?: string) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      if (referenceText) {
        formData.append('reference_text', referenceText);
      }

      const response = await fetch(`${BACKEND_URL}/api/analyze-audio`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze audio');
      }

      const result = await response.json();
      setMetrics(prev => ({ ...prev, audio: result }));
      return result;
    } catch (error) {
      console.error('Audio analysis error:', error);
      setMetrics(prev => ({ ...prev, error: 'Audio analysis failed' }));
      throw error;
    }
  }, []);

  /**
   * Start real-time audio capture (experimental)
   * Note: For better results, use sendAudioForAnalysis with recorded audio
   */
  const startAudioCapture = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      
      // Create script processor for audio data access
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioBufferRef.current.push(new Float32Array(inputData));
        
        // Send audio every 3 seconds
        const now = Date.now();
        if (now - lastAudioSendRef.current >= 3000 && audioBufferRef.current.length > 0) {
          // Combine buffer
          const totalLength = audioBufferRef.current.reduce((sum, arr) => sum + arr.length, 0);
          const combined = new Float32Array(totalLength);
          let offset = 0;
          audioBufferRef.current.forEach(arr => {
            combined.set(arr, offset);
            offset += arr.length;
          });
          
          // Convert to WAV blob and send
          const wavBlob = float32ToWav(combined, 16000);
          sendAudioForAnalysis(wavBlob);
          
          audioBufferRef.current = [];
          lastAudioSendRef.current = now;
        }
      };
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      audioContextRef.current = audioContext;
      processorRef.current = processor;
      
      setMetrics(prev => ({ ...prev, isAudioActive: true }));
      
      return stream;
    } catch (error) {
      console.error('Failed to start audio:', error);
      setMetrics(prev => ({ ...prev, error: 'Microphone access denied' }));
      throw error;
    }
  }, [sendAudioForAnalysis]);

  /**
   * Stop audio capture
   */
  const stopAudioCapture = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    audioBufferRef.current = [];
    setMetrics(prev => ({ ...prev, isAudioActive: false }));
  }, []);

  /**
   * Clean up on unmount
   */
  useEffect(() => {
    return () => {
      stopVideo();
      stopAudioCapture();
    };
  }, [stopVideo, stopAudioCapture]);

  return {
    metrics,
    startVideo,
    stopVideo,
    startAudioCapture,
    stopAudioCapture,
    sendAudioForAnalysis,
  };
}

/**
 * Convert Float32Array audio data to WAV blob
 */
function float32ToWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  
  // Convert samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

export default useInterviewMetrics;
