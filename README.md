# Student Distraction Detection System Using YOLOv8 and MediaPipe Face Mesh

Real-time attention feedback for one student, using a laptop webcam, a pre-trained YOLOv8n model (phone detection)
and MediaPipe Face Mesh (eye aspect ratio and head yaw). A Flask server streams the annotated video and a JSON
status to a browser dashboard. Runs on the CPU; no video or images are saved.

Bhat Bikram Singh (ID 2317146), Department of Smart Computing, Kyungdong University Global.

## States
Focused, Distracted (head turned, |yaw| > 13), Drowsy (EAR < 0.25 for 20 consecutive frames), Phone Use
(YOLOv8n class "cell phone", confidence > 0.70). "Please face the camera" is shown when no face is found.

## Run
1. Install Python 3.12 and create a virtual environment.
2. `pip install -r requirements.txt` (use a CPU build of PyTorch if you have no GPU).
3. Folder layout: `server.py`, `templates/index.html`, `static/` (university logo).
4. `python server.py`, then open http://127.0.0.1:146 in a browser.
The first start downloads `yolov8n.pt` automatically (internet needed once).

## Evaluation (thesis, Chapter 5)
240 trials, 3 participants x 4 behaviours x 20: 226 correct (94.17%); about 11-13 FPS on a laptop CPU.
