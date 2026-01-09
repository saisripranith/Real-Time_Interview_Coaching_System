import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
smoothed_angles = deque(maxlen=5)  # short smoothing window

# def rotationMatrixToEulerAngles(R):
#     # R: 3x3 rotation matrix -> returns pitch, yaw, roll (degrees)
#     sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
#     singular = sy < 1e-6
#     if not singular:
#         x = math.atan2(R[2,1], R[2,2])
#         y = math.atan2(-R[2,0], sy)
#         z = math.atan2(R[1,0], R[0,0])
#     else:
#         x = math.atan2(-R[1,2], R[1,1])
#         y = math.atan2(-R[2,0], sy)
#         z = 0
#     # Convert to degrees: return yaw (y), pitch (x), roll (z) if you prefer that order.
#     return math.degrees(y), math.degrees(x), math.degrees(z)

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])   # pitch
        y = math.atan2(-R[2,0], sy)      # yaw
        z = math.atan2(R[1,0], R[0,0])   # roll
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # Convert radians → degrees
    yaw, pitch, roll = map(math.degrees, [y, -x, z])  # <--- invert pitch
    # Normalize to [-180, 180]
    yaw = (yaw + 180) % 360 - 180
    pitch = (pitch + 180) % 360 - 180
    roll = (roll + 180) % 360 - 180
    return yaw, pitch, roll

# print(f"Yaw: {s_yaw:.1f}  Pitch: {s_pitch:.1f}  Roll: {s_roll:.1f}")


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("camera frame failure")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # NOTE: compute pose using the non-flipped coordinates to avoid mirror confusion.
    results = face_mesh.process(rgb)

    display = frame.copy()  # draw on this
    if not results.multi_face_landmarks:
        # No face detected - add message
        cv2.putText(display, "❗ Please look at the camera!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        face = results.multi_face_landmarks[0]

        # Useful MediaPipe indices (stable points)
        idx_nose = 1         # nose tip
        idx_chin = 152       # chin
        idx_left_eye = 33    # left eye outer
        idx_right_eye = 263  # right eye outer
        idx_mouth_left = 61
        idx_mouth_right = 291

        # get 2D image points in pixel coords
        def lm2xy(i):
            lm = face.landmark[i]
            return np.array([lm.x * w, lm.y * h], dtype=np.float64)

        image_points = np.array([
            lm2xy(idx_nose),       # nose tip
            lm2xy(idx_chin),       # chin
            lm2xy(idx_left_eye),   # left eye outer
            lm2xy(idx_right_eye),  # right eye outer
            lm2xy(idx_mouth_left), # left mouth corner
            lm2xy(idx_mouth_right) # right mouth corner
        ], dtype=np.float64)

        # 3D model points (generic face model coordinates in mm)
        model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -63.6, -12.5],      # Chin
            [-43.3, 32.7, -26.0],     # Left eye outer
            [43.3, 32.7, -26.0],      # Right eye outer
            [-28.9, -28.9, -24.1],    # Left mouth corner
            [28.9, -28.9, -24.1]      # Right mouth corner
        ], dtype=np.float64)

        # Camera internals
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))  # assuming no lens distortion

        # Solve PnP
        success_pnp, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success_pnp:
            R_mat, _ = cv2.Rodrigues(rotation_vec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(R_mat)  # yaw, pitch, roll degrees

            # smoothing (simple moving average)
            smoothed_angles.append((yaw, pitch, roll))
            s_yaw = np.mean([a[0] for a in smoothed_angles])
            s_pitch = np.mean([a[1] for a in smoothed_angles])
            s_roll = np.mean([a[2] for a in smoothed_angles])

            print(f"Yaw: {s_yaw:.1f}  Pitch: {s_pitch:.1f}  Roll: {s_roll:.1f}")

            # Decide looking or not. Positive/negative sign depends on coordinate conventions.
            # Typical convention: yaw ~ positive when looking to camera's left (subject right).
            yaw_abs = abs(s_yaw)
            # pitch_mod = abs((abs(s_pitch) - 180))  # normalize pitch wraparound
            # Normalize pitch cyclically around ±180
            pitch_mod = min(abs(s_pitch), abs(abs(s_pitch) - 180))

            looking = yaw_abs < 25 and pitch_mod < 20
# status = "✅ Looking at camera" if looking else "⚠️ Not looking at camera"


            status = "✅ Looking at camera" if looking else "⚠️ Not looking at camera"
            color = (0,255,0) if looking else (0,0,255)

            cv2.putText(display, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, f"Yaw: {yaw_abs:.1f}  Pitch: {pitch_mod:.1f}  Roll: {s_roll:.1f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # draw the selected image_points for debug
            for p in image_points:
                cv2.circle(display, (int(p[0]), int(p[1])), 3, (255,255,0), -1)

    # OPTIONAL: flip for selfie display after pose computed (so pose used non-flipped coords)
    display = cv2.flip(display, 1)

    cv2.imshow("Head Pose (solvePnP)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
