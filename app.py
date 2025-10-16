import os
import cv2
import time
import threading
import numpy as np
import yaml
import pickle
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Custom modules
import models
import database
from train_model import train_embeddings  # trainer function you created

# --- CONFIG ---
KNOWN_FACES_DIR = "known_faces"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"

DETECTOR_CONFIDENCE = 0.6
RECOGNITION_THRESHOLD = 0.45      # cosine similarity threshold
CONFIRMATION_BUFFER_SIZE = 3      # smaller buffer for group detection
TRACK_MAX_MISS_SEC = 2.0          # remove stale tracks
CENTER_MATCH_DIST = 80            # pixels to match centroids between frames
ENHANCE = True                    # toggle enhancement on/off
ATTENDANCE_SESSION = {}           # name -> last marked timestamp (for duplicate handling)

# --- INIT ---
app = Flask(__name__)
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Video capture
video_capture = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

# Thread-safe embeddings
emb_lock = threading.Lock()
known_embeddings = np.array([])
known_names = []

# Recognition state
tracks = {}  # track_id -> { 'center': (x,y), 'buf': [names], 'last_seen': ts, 'last_marked': ts }
next_track_id = 0
tracks_lock = threading.Lock()

# DB init
models.initialize_database()

# InsightFace
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))


def load_embeddings_labels():
    """Load embeddings and labels into memory (thread-safe)."""
    global known_embeddings, known_names
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        try:
            with open(MODEL_FILE, "r") as f:
                data = yaml.safe_load(f)
            with open(LABELS_FILE, "rb") as f:
                labels_mapping = pickle.load(f)
            embeddings = np.array(data.get("embeddings", []), dtype=np.float32)
            ids = data.get("ids", [])
            names = [labels_mapping[i] for i in ids]
            with emb_lock:
                known_embeddings = embeddings
                known_names = names
            print(f"[INFO] Loaded {len(known_names)} embeddings.")
        except Exception as e:
            print(f"[ERROR] load_embeddings_labels: {e}")
    else:
        with emb_lock:
            known_embeddings = np.array([])
            known_names = []
        print("[INFO] No embeddings/labels found. Running with empty DB.")


# Initial load
load_embeddings_labels()


# -----------------------
# Enhancement utilities
# -----------------------
def enhance_image(img):
    """Enhance brightness/contrast and sharpen for low-light conditions."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        img_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_sharp = cv2.filter2D(img_clahe, -1, kernel)
        return img_sharp
    except Exception:
        return img


# -----------------------
# Tracking utilities
# -----------------------
def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def match_track(center):
    with tracks_lock:
        best_id = None
        best_dist = None
        for tid, info in tracks.items():
            cx, cy = info['center']
            dist = np.hypot(center[0] - cx, center[1] - cy)
            if dist <= CENTER_MATCH_DIST and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_id = tid
        return best_id


def create_track(center):
    global next_track_id
    with tracks_lock:
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = {'center': center, 'buf': [], 'last_seen': time.time(), 'last_marked': 0}
        return tid


def cleanup_tracks():
    now = time.time()
    with tracks_lock:
        stale = [tid for tid, info in tracks.items() if (now - info['last_seen']) > TRACK_MAX_MISS_SEC]
        for tid in stale:
            del tracks[tid]


# -----------------------
# Recognition thread
# -----------------------
def recognize_loop():
    global ATTENDANCE_SESSION
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            frame = current_frame.copy()

        proc = enhance_image(frame) if ENHANCE else frame

        try:
            faces = face_app.get(proc)
        except Exception as e:
            print(f"[ERROR] InsightFace get() failed: {e}")
            faces = []

        with emb_lock:
            emb_local = known_embeddings.copy()
            names_local = known_names.copy()

        detections_info = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            center = center_of_bbox(bbox)
            embedding = f.embedding.reshape(1, -1)
            detections_info.append((center, bbox, embedding))

        for center, bbox, embedding in detections_info:
            tid = match_track(center)
            if tid is None:
                tid = create_track(center)

            with tracks_lock:
                tracks[tid]['center'] = center
                tracks[tid]['last_seen'] = time.time()

            # Recognition
            name = "Unknown"
            score = 0.0
            if emb_local.size > 0:
                try:
                    sims = cosine_similarity(embedding, emb_local)[0]
                    best_idx = int(np.argmax(sims))
                    score = float(sims[best_idx])
                    if score > RECOGNITION_THRESHOLD:
                        name = names_local[best_idx]
                except Exception:
                    name = "Unknown"

            # Track buffer
            with tracks_lock:
                buf = tracks[tid]['buf']
                buf.append(name)
                if len(buf) > CONFIRMATION_BUFFER_SIZE:
                    buf.pop(0)
                tracks[tid]['buf'] = buf

                # Confirm attendance
                if len(buf) == CONFIRMATION_BUFFER_SIZE and len(set(buf)) == 1 and buf[0] != "Unknown":
                    confirmed_name = buf[0]
                    now = time.time()
                    # Only mark once per session (duplicate prevention)
                    last_marked = ATTENDANCE_SESSION.get(confirmed_name, 0)
                    if (now - last_marked) > 10:  # 10s debounce
                        db_conn = database.create_db_connection()
                        if db_conn:
                            try:
                                database.mark_attendance(db_conn, confirmed_name)
                                ATTENDANCE_SESSION[confirmed_name] = now
                                print(f"[ATTENDANCE] Marked {confirmed_name}")
                            except Exception as e:
                                print(f"[ERROR] mark_attendance failed: {e}")
                            finally:
                                db_conn.close()
                        tracks[tid]['last_marked'] = now

        cleanup_tracks()
        time.sleep(0.08)


# Start recognition thread
recognition_thread = threading.Thread(target=recognize_loop, daemon=True)
recognition_thread.start()


# -----------------------
# Video streaming with summary overlay
# -----------------------
def generate_frames():
    global current_frame
    while True:
        success, frame = video_capture.read()
        if not success:
            time.sleep(0.02)
            continue

        display = frame.copy()

        # Draw tracks
        with tracks_lock:
            for tid, info in tracks.items():
                cx, cy = info['center']
                x1, y1 = max(0, cx - 60), max(0, cy - 60)
                x2, y2 = min(display.shape[1] - 1, cx + 60), min(display.shape[0] - 1, cy + 60)
                label = "Unknown"
                if info['buf']:
                    label = max(set(info['buf']), key=info['buf'].count)
                    if label != "Unknown":
                        if len(info['buf']) == CONFIRMATION_BUFFER_SIZE and len(set(info['buf'])) == 1:
                            label = f"{label} (Confirmed)"
                        else:
                            label = f"{label} (Confirming...)"
                color = (0, 255, 0) if "Confirmed" in label else (0, 255, 255) if "Confirming" in label else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Live attendance summary
        present_count = len(ATTENDANCE_SESSION)
        with emb_lock:
            total_users = len(known_names)
        cv2.putText(display, f"Present: {present_count}/{total_users}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Enhancement status
        if ENHANCE:
            cv2.putText(display, "Enhance: ON", (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        with frame_lock:
            current_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', display)
        if not ret:
            time.sleep(0.02)
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -----------------------
# Web routes (manage/reports unchanged)
# -----------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manage', methods=['GET', 'POST'])
def manage_users():
    db_conn = database.create_db_connection()
    if not db_conn:
        return "Database connection failed", 500

    if request.method == 'POST':
        name = request.form.get('name')
        photo = request.files.get('photo')
        if name and photo:
            safe_name = secure_filename(name.replace(" ", "_"))
            extension = os.path.splitext(photo.filename)[1] or '.jpg'
            filepath = os.path.join(KNOWN_FACES_DIR, safe_name + extension)
            try:
                photo.save(filepath)
            except Exception as e:
                print(f"[ERROR] save failed: {e}")
                db_conn.close()
                return "File save failed", 500
            try:
                database.get_or_create_user(db_conn, name)
            except Exception as e:
                print(f"[ERROR] get_or_create_user: {e}")
            try:
                print("[TRAINER] Retraining embeddings due to new user...")
                ok = train_embeddings()
                if ok:
                    load_embeddings_labels()
                else:
                    print("[TRAINER] Retrain failed; keeping old embeddings.")
            except Exception as e:
                print(f"[ERROR] retrain failed: {e}")
            db_conn.close()
            return redirect(url_for('manage_users'))

    users = database.get_all_users(db_conn)
    db_conn.close()
    return render_template('manage_users.html', users=users)


@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user_route(user_id):
    db_conn = database.create_db_connection()
    if not db_conn:
        return "Database connection failed", 500

    users = database.get_all_users(db_conn)
    user_to_delete = next((u for u in users if u['id'] == user_id), None)
    if user_to_delete:
        safe_name = secure_filename(user_to_delete['name'].replace(" ", "_"))
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(KNOWN_FACES_DIR, safe_name + ext)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[WARN] remove file failed: {e}")
                break
        try:
            database.delete_user(db_conn, user_id)
        except Exception as e:
            print(f"[ERROR] delete_user failed: {e}")
        try:
            print("[TRAINER] Retraining embeddings after deletion...")
            ok = train_embeddings()
            if ok:
                load_embeddings_labels()
            else:
                print("[TRAINER] Retrain failed; keeping old embeddings.")
        except Exception as e:
            print(f"[ERROR] retrain failed: {e}")
    db_conn.close()
    return redirect(url_for('manage_users'))


@app.route('/reports', methods=['GET'])
def reports():
    db_conn = database.create_db_connection()
    if not db_conn:
        return "Database connection failed", 500

    date_str = request.args.get('date', datetime.today().strftime('%Y-%m-%d'))
    report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    all_users = database.get_all_users(db_conn)
    present_today = database.get_attendance_for_date(db_conn, report_date)
    db_conn.close()

    present_names = {record[0] for record in present_today}
    present_timestamps = {record[0]: record[1].strftime('%I:%M:%S %p') for record in present_today}

    report_data = []
    for user in all_users:
        user_name = user['name']
        status = "Present" if user_name in present_names else "Absent"
        timestamp = present_timestamps.get(user_name)
        report_data.append({
            'name': user_name,
            'role': user['role'],
            'status': status,
            'timestamp': timestamp
        })
    return render_template('reports.html', report_data=report_data, report_date=date_str)


@app.route('/get_attendance')
def get_attendance():
    db_conn = database.create_db_connection()
    if not db_conn:
        return jsonify([])

    attendance_today = database.get_todays_attendance(db_conn)
    db_conn.close()

    formatted_attendance = []
    for name, timestamp in attendance_today:
        formatted_attendance.append({
            "name": name,
            "time": timestamp.strftime('%I:%M:%S %p')
        })
    return jsonify(formatted_attendance)


# --- MAIN ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
