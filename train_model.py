# trainer.py
import os
import cv2
import numpy as np
import pickle
import yaml
from insightface.app import FaceAnalysis

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"


def train_embeddings():
    print("[TRAINER] Generating embeddings using InsightFace...")

    # Initialize InsightFace model
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = []
    label_ids = {}   # name -> id mapping
    current_id = 0
    ids = []

    # Process each image in known_faces
    for file in os.listdir(KNOWN_FACES_DIR):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(KNOWN_FACES_DIR, file)
        name = os.path.splitext(file)[0].replace("_", " ")

        img = cv2.imread(path)
        if img is None:
            print(f"[SKIP] Could not read {file}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"[SKIP] No face detected in {file}")
            continue

        embedding = faces[0].embedding
        embeddings.append(embedding)

        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1
        ids.append(label_ids[name])

        print(f"[OK] Processed {name}")

    if len(embeddings) == 0:
        print("[ERROR] No embeddings generated. Check your images.")
        return False

    # Save embeddings to YAML
    with open(MODEL_FILE, "w") as f:
        yaml.dump({
            "embeddings": np.array(embeddings).tolist(),
            "ids": ids
        }, f, default_flow_style=False)

    # Save labels mapping (id -> name) to Pickle
    labels_to_save = {v: k for k, v in label_ids.items()}
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(labels_to_save, f)

    print(f"[DONE] Saved {len(label_ids)} embeddings to {MODEL_FILE}")
    return True


if __name__ == "__main__":
    train_embeddings()
