"""
Head Pose Detection Module
Optimized version of the face tracking for integration with the backend server.
Provides real-time head pose estimation (yaw, pitch, roll) and eye contact detection.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import base64

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class HeadPoseResult:
    """Results from head pose analysis"""
    yaw: float  # Left/Right rotation in degrees
    pitch: float  # Up/Down rotation in degrees
    roll: float  # Tilt rotation in degrees
    is_looking_at_camera: bool
    attention_score: float  # 0-100
    eye_contact_score: float  # 0-100
    face_detected: bool
    landmarks: Optional[List[Tuple[int, int]]] = None
    
    def to_dict(self):
        return {
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "is_looking_at_camera": bool(self.is_looking_at_camera),
            "attention_score": round(self.attention_score, 1),
            "eye_contact_score": round(self.eye_contact_score, 1),
            "face_detected": bool(self.face_detected)
        }


class HeadPoseDetector:
    """
    Detects head pose and calculates eye contact metrics using MediaPipe Face Mesh.
    """
    
    # MediaPipe Face Mesh landmark indices
    IDX_NOSE = 1
    IDX_CHIN = 152
    IDX_LEFT_EYE = 33
    IDX_RIGHT_EYE = 263
    IDX_MOUTH_LEFT = 61
    IDX_MOUTH_RIGHT = 291
    
    # 3D model points for a generic face (in mm)
    MODEL_POINTS = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -63.6, -12.5],      # Chin
        [-43.3, 32.7, -26.0],     # Left eye outer
        [43.3, 32.7, -26.0],      # Right eye outer
        [-28.9, -28.9, -24.1],    # Left mouth corner
        [28.9, -28.9, -24.1]      # Right mouth corner
    ], dtype=np.float64)
    
    # Thresholds for determining if looking at camera
    YAW_THRESHOLD = 25  # degrees
    PITCH_THRESHOLD = 20  # degrees
    
    def __init__(self, smoothing_window: int = 5):
        """
        Initialize the head pose detector.
        
        Args:
            smoothing_window: Number of frames to average for smoothing
        """
        # Initialize MediaPipe Face Mesh with tuned thresholds for reliability on varied lighting/cameras
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        
        # Smoothing buffers
        self.smoothed_angles = deque(maxlen=smoothing_window)
        self.attention_history = deque(maxlen=30)  # ~1 second at 30fps
        self.eye_contact_history = deque(maxlen=30)
        
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (yaw, pitch, roll).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])   # pitch
            y = math.atan2(-R[2, 0], sy)        # yaw
            z = math.atan2(R[1, 0], R[0, 0])   # roll
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # Convert radians to degrees and adjust signs
        yaw, pitch, roll = map(math.degrees, [y, -x, z])
        
        # Normalize to [-180, 180]
        yaw = (yaw + 180) % 360 - 180
        pitch = (pitch + 180) % 360 - 180
        roll = (roll + 180) % 360 - 180
        
        return yaw, pitch, roll
    
    def _get_image_points(self, face_landmarks, width: int, height: int) -> np.ndarray:
        """Extract 2D image points from face landmarks"""
        def lm_to_xy(idx):
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * width, lm.y * height], dtype=np.float64)
        
        return np.array([
            lm_to_xy(self.IDX_NOSE),
            lm_to_xy(self.IDX_CHIN),
            lm_to_xy(self.IDX_LEFT_EYE),
            lm_to_xy(self.IDX_RIGHT_EYE),
            lm_to_xy(self.IDX_MOUTH_LEFT),
            lm_to_xy(self.IDX_MOUTH_RIGHT)
        ], dtype=np.float64)
    
    def _calculate_attention_score(self, yaw_abs: float, pitch_mod: float) -> float:
        """
        Calculate attention score based on head pose.
        
        Args:
            yaw_abs: Absolute yaw angle
            pitch_mod: Modified pitch angle
            
        Returns:
            Attention score from 0 to 100
        """
        # Perfect attention when looking straight at camera
        yaw_penalty = min(yaw_abs / self.YAW_THRESHOLD, 1.0) * 50
        pitch_penalty = min(pitch_mod / self.PITCH_THRESHOLD, 1.0) * 50
        
        score = 100 - yaw_penalty - pitch_penalty
        return max(0, min(100, score))
    
    def _calculate_eye_contact_score(self, is_looking: bool) -> float:
        """
        Calculate running eye contact score based on history.
        
        Args:
            is_looking: Whether currently looking at camera
            
        Returns:
            Eye contact score from 0 to 100
        """
        self.eye_contact_history.append(1.0 if is_looking else 0.0)
        
        if len(self.eye_contact_history) == 0:
            return 50.0
            
        return (sum(self.eye_contact_history) / len(self.eye_contact_history)) * 100
    
    def process_frame(self, frame: np.ndarray) -> HeadPoseResult:
        """
        Process a single video frame and return head pose analysis.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            HeadPoseResult with all metrics
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                is_looking_at_camera=False,
                attention_score=0.0,
                eye_contact_score=self._calculate_eye_contact_score(False),
                face_detected=False
            )
        
        face = results.multi_face_landmarks[0]
        
        # Get 2D image points
        image_points = self._get_image_points(face, w, h)
        
        # Camera intrinsics (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP to get head pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                is_looking_at_camera=False,
                attention_score=0.0,
                eye_contact_score=self._calculate_eye_contact_score(False),
                face_detected=True
            )
        
        # Convert rotation vector to matrix and then to Euler angles
        R_mat, _ = cv2.Rodrigues(rotation_vec)
        yaw, pitch, roll = self._rotation_matrix_to_euler_angles(R_mat)
        
        # Smooth the angles
        self.smoothed_angles.append((yaw, pitch, roll))
        s_yaw = np.mean([a[0] for a in self.smoothed_angles])
        s_pitch = np.mean([a[1] for a in self.smoothed_angles])
        s_roll = np.mean([a[2] for a in self.smoothed_angles])
        
        # Calculate metrics
        yaw_abs = abs(s_yaw)
        pitch_mod = min(abs(s_pitch), abs(abs(s_pitch) - 180))
        
        is_looking = yaw_abs < self.YAW_THRESHOLD and pitch_mod < self.PITCH_THRESHOLD
        attention_score = self._calculate_attention_score(yaw_abs, pitch_mod)
        eye_contact_score = self._calculate_eye_contact_score(is_looking)
        
        # Get landmark points for visualization
        landmarks = [(int(p[0]), int(p[1])) for p in image_points]
        
        return HeadPoseResult(
            yaw=s_yaw,
            pitch=s_pitch,
            roll=s_roll,
            is_looking_at_camera=is_looking,
            attention_score=attention_score,
            eye_contact_score=eye_contact_score,
            face_detected=True,
            landmarks=landmarks
        )
    
    def process_base64_frame(self, base64_data: str) -> HeadPoseResult:
        """
        Process a base64-encoded image frame.
        
        Args:
            base64_data: Base64-encoded image data (may include data URI prefix)
            
        Returns:
            HeadPoseResult
        """
        # Remove data URI prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                is_looking_at_camera=False,
                attention_score=0.0,
                eye_contact_score=0.0,
                face_detected=False
            )
        
        return self.process_frame(frame)
    
    def draw_annotations(self, frame: np.ndarray, result: HeadPoseResult) -> np.ndarray:
        """
        Draw head pose annotations on a frame.
        
        Args:
            frame: Original frame
            result: HeadPoseResult from process_frame
            
        Returns:
            Annotated frame
        """
        display = frame.copy()
        
        if not result.face_detected:
            cv2.putText(
                display, "No face detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            return display
        
        # Status text
        status = "Looking at camera" if result.is_looking_at_camera else "Not looking at camera"
        color = (0, 255, 0) if result.is_looking_at_camera else (0, 0, 255)
        
        cv2.putText(
            display, status,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        
        # Angles
        cv2.putText(
            display,
            f"Yaw: {abs(result.yaw):.1f} Pitch: {abs(result.pitch):.1f} Roll: {result.roll:.1f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        # Scores
        cv2.putText(
            display,
            f"Attention: {result.attention_score:.0f}% Eye Contact: {result.eye_contact_score:.0f}%",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        
        # Draw landmarks
        if result.landmarks:
            for point in result.landmarks:
                cv2.circle(display, point, 3, (255, 255, 0), -1)
        
        return display
    
    def cleanup(self):
        """Release resources"""
        self.face_mesh.close()


# Singleton instance
_detector_instance: Optional[HeadPoseDetector] = None


def get_head_pose_detector(smoothing_window: int = 5) -> HeadPoseDetector:
    """Get or create the head pose detector singleton"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HeadPoseDetector(smoothing_window)
    return _detector_instance


if __name__ == "__main__":
    # Test the head pose detector with webcam
    print("Testing Head Pose Detector...")
    
    detector = HeadPoseDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        result = detector.process_frame(frame)
        annotated = detector.draw_annotations(frame, result)
        
        # Flip for selfie view
        annotated = cv2.flip(annotated, 1)
        
        cv2.imshow("Head Pose Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
