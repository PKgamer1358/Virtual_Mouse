
import csv
import os
import sys
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from custom_gestures import DATA_FILE, ALLOWED_LABELS

# Save candidate separately. Do not overwrite the original model.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_MODEL_FILE = os.path.join(
    BASE_DIR,
    "custom_gesture_model_candidate.pkl"
)

# Check dataset
if not os.path.exists(DATA_FILE):
    print("Dataset not found:", DATA_FILE)
    sys.exit(1)

# Load dataset
with open(DATA_FILE, "r", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("Dataset is empty.")
    sys.exit(1)

labels = []
features = []
invalid_rows = 0

# Extract 63 features for every valid sample.
for row in rows:
    label = row.get("label", "").strip().upper()

    if label not in ALLOWED_LABELS:
        invalid_rows += 1
        continue

    try:
        vector = [
            float(row["f" + str(i)])
            for i in range(63)
        ]

        if not np.all(np.isfinite(vector)):
            raise ValueError("Non-finite feature")

        labels.append(label)
        features.append(vector)

    except (KeyError, ValueError, TypeError):
        invalid_rows += 1

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels)

if len(y) == 0:
    print("No valid samples found.")
    sys.exit(1)

classes, counts = np.unique(y, return_counts=True)

print("\n====================================")
print("GESTURE MODEL CANDIDATE TRAINING")
print("====================================")

print("\nTotal valid samples:", len(y))
print("Invalid rows:", invalid_rows)
print("Class distribution:")

for name, count in zip(classes, counts):
    print("{}: {}".format(name, count))

if len(classes) < 2 or np.min(counts) < 2:
    print("Insufficient samples for training.")
    sys.exit(1)

# Create a stratified diagnostic split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1
)

# Diagnostic training
print("\nTraining diagnostic model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nDIAGNOSTIC HOLDOUT ACCURACY")
print("---------------------------")

print(
    "Accuracy: {:.2f}%".format(
        accuracy_score(y_test, predictions) * 100
    )
)

print("\nCLASSIFICATION REPORT")
print("---------------------------")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print("---------------------------")

print("Class order:", list(model.classes_))

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )
)

# Train the final candidate on all valid training samples.
print("\nTraining final candidate on all valid samples...")

candidate_model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1
)

candidate_model.fit(X, y)

# Save to a different file.
joblib.dump(candidate_model, CANDIDATE_MODEL_FILE)

print("\n====================================")
print("CANDIDATE TRAINING COMPLETE")
print("====================================")

print("Candidate model:", CANDIDATE_MODEL_FILE)
print("Training samples:", len(y))
print("Classes:", list(candidate_model.classes_))

print("\nOriginal model was NOT modified.")
