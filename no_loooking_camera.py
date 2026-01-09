import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from camera")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect faces
    result = face_mesh.process(rgb)
    
    # Draw the face mesh annotations on the frame
    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            # Draw the face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Get landmarks for eyes and nose tip
            nose = face.landmark[1]
            left_eye = face.landmark[33]
            right_eye = face.landmark[263]
            
            # Calculate yaw (left-right head rotation)
            yaw = math.degrees(math.atan2(right_eye.x - left_eye.x, right_eye.y - left_eye.y))
            
            # Draw status text
            status = "✅ Looking at camera" if abs(yaw) <= 85 else "⚠️ Not looking at camera"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                      (0, 255, 0) if abs(yaw) <= 85 else (0, 0, 255), 2)

            # Draw yaw angle
            cv2.putText(frame, f"Yaw: {yaw:.1f} degrees", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Face Mesh', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()