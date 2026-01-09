import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
# from fer import FER  # pip install fer

# ------------------- MediaPipe & FER Setup -------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Emotion detector
# emotion_detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)

# Smoothing windows
smoothed_angles = deque(maxlen=5)
blink_history = deque(maxlen=150)  # ~5 sec at 30fps
attention_history = deque(maxlen=150)

# Eye aspect ratio indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# ------------------- Helper Functions -------------------
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    yaw, pitch, roll = map(math.degrees, [y, -x, z])
    yaw = (yaw + 180) % 360 - 180
    pitch = (pitch + 180) % 360 - 180
    roll = (roll + 180) % 360 - 180
    return yaw, pitch, roll

def lm2xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)

def eye_aspect_ratio(eye_landmarks):
    # eye_landmarks: array of 6 (x,y) points
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ------------------- Main Loop -------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame first
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    display = frame.copy()

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        # ---------------- Head Pose ----------------
        idx_nose, idx_chin = 1, 152
        idx_left_eye, idx_right_eye = 33, 263
        idx_mouth_left, idx_mouth_right = 61, 291

        image_points = np.array([
            lm2xy(face.landmark[idx_nose], w, h),
            lm2xy(face.landmark[idx_chin], w, h),
            lm2xy(face.landmark[idx_left_eye], w, h),
            lm2xy(face.landmark[idx_right_eye], w, h),
            lm2xy(face.landmark[idx_mouth_left], w, h),
            lm2xy(face.landmark[idx_mouth_right], w, h)
        ], dtype=np.float64)

        model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1]
        ], dtype=np.float64)

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))
        success_pnp, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if success_pnp:
            R_mat, _ = cv2.Rodrigues(rotation_vec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(R_mat)
            smoothed_angles.append((yaw, pitch, roll))
            s_yaw = np.mean([a[0] for a in smoothed_angles])
            s_pitch = np.mean([a[1] for a in smoothed_angles])
            s_roll = np.mean([a[2] for a in smoothed_angles])

            yaw_abs = abs(s_yaw)
            pitch_mod = min(abs(s_pitch), abs(abs(s_pitch) - 180))
            looking = yaw_abs < 25 and pitch_mod < 20
            attention_history.append(1 if looking else 0)

            status = "✅ Looking at camera" if looking else "⚠️ Not looking"
            color = (0,255,0) if looking else (0,0,255)
            cv2.putText(display, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            cv2.putText(display, f"Yaw:{yaw_abs:.1f} Pitch:{pitch_mod:.1f} Roll:{s_roll:.1f}", 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # ---------------- Blink Detection ----------------
        left_eye = np.array([lm2xy(face.landmark[i], w, h) for i in LEFT_EYE_IDX])
        right_eye = np.array([lm2xy(face.landmark[i], w, h) for i in RIGHT_EYE_IDX])
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        blink_history.append(ear)
        blink_thresh = 0.20
        blink = ear < blink_thresh
        cv2.putText(display, f"Blink:{blink}", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        # ---------------- Emotion Detection ----------------
        face_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        # emotions = emotion_detector.detect_emotions(face_rgb)
        dominant_emotion = "Neutral"
        # if emotions:
            # dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        cv2.putText(display, f"Emotion:{dominant_emotion}", (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # ---------------- Confidence Score ----------------
        attention_score = np.mean(attention_history) if attention_history else 0
        blink_score = 1.0 - np.mean([1 if b < blink_thresh else 0 for b in blink_history])  # high blink -> stress
        emotion_score = 1.0 if dominant_emotion in ["Happy","Neutral"] else 0.5  # basic
        confidence = (attention_score + blink_score + emotion_score)/3
        cv2.putText(display, f"Confidence:{confidence*100:.1f}%", (10,150), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    else:
        cv2.putText(display, "❗ Please look at the camera!", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Interview Coach", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
