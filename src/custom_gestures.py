
import os
import csv
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "custom_gesture_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "custom_gesture_model.pkl")

# Only these labels can be mapped to existing mouse-control actions.
ALLOWED_LABELS = {
    "PALM", "FIST", "INDEX", "FIRST2", "V_GEST"
}


def extract_features(hand_landmarks):
    """
    Convert 21 MediaPipe landmarks into 63 normalized features.
    Coordinates are relative to the wrist and scale-normalized.
    """
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Move the coordinate origin to the wrist.
    points -= points[0]

    # Normalize for different hand sizes and camera distances.
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-6:
        points /= scale

    return points.flatten().tolist()


def load_model():
    """Return the trained model, or None if it has not been trained."""
    if not os.path.exists(MODEL_FILE):
        return None

    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        print("Could not load custom gesture model:", exc)
        return None
