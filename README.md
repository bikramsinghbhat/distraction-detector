# Student Distraction Detection System

Developed by: Bhat Bikram Singh (2317146)
Department: Smart Computing, Kyungdong University Global
Supervisor: Prof. Mohammed A.A

## What It Detects
- Focused — looking at screen
- Distracted — looking away
- Drowsy — eyes closing
- Phone Use — phone in frame

## Tech Stack
- YOLOv8 — phone detection
- MediaPipe Face Mesh — facial landmarks
- OpenCV — video processing
- Flask — backend server
- HTML/CSS/JavaScript — web dashboard
- Chart.js — attention chart

## How To Run
1. Create virtual environment: python -m venv venv
2. Activate: venv\Scripts\activate
3. Install libraries: pip install -r requirements.txt
4. Run: python server.py
5. Open browser: http://127.0.0.1:5000