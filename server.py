from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load models
model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Thresholds
EAR_THRESHOLD = 0.25
DROWSY_FRAMES = 20
YAW_THRESHOLD = 13
PITCH_THRESHOLD = 13
frame_counter = 0
current_status = "Loading..."

def calculate_EAR(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, w, h):
    face_3d, face_2d = [], []
    for idx in [1, 9, 57, 130, 287, 359]:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[1] * 360, angles[0] * 360

def detect_phone(frame):
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 67 and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Phone {conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return True
    return False

def generate_frames():
    global frame_counter, current_status
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        phone_detected = detect_phone(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status = "No Face Detected"
        color = (0, 0, 255)

        if phone_detected:
            status = "PHONE DETECTED!"
            color = (255, 0, 0)
        elif results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in LEFT_EYE])
            right_eye = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in RIGHT_EYE])
            avg_EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
            yaw, pitch = get_head_pose(landmarks, w, h)

            if avg_EAR < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= DROWSY_FRAMES:
                    status = "DROWSY!"
                    color = (0, 0, 255)
            elif abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
                frame_counter = 0
                status = "DISTRACTED!"
                color = (0, 165, 255)
            else:
                frame_counter = 0
                status = "Focused"
                color = (0, 255, 0)

        current_status = status

        # Draw status on frame
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {'status': current_status}

if __name__ == '__main__':
    app.run(debug=False)