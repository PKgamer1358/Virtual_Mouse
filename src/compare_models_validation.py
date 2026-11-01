
import os
import csv
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_MODEL = os.path.join(
    BASE_DIR, "custom_gesture_model_original.pkl"
)

CANDIDATE_MODEL = os.path.join(
    BASE_DIR, "custom_gesture_model_candidate.pkl"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR, "custom_gesture_validation.csv"
)

# Check required files
for path in [ORIGINAL_MODEL, CANDIDATE_MODEL, VALIDATION_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError("File not found: " + path)

# Load validation data
features = []
labels = []
invalid_rows = 0

with open(VALIDATION_FILE, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            label = row["label"].strip().upper()
            vector = [
                float(row["f" + str(i)])
                for i in range(63)
            ]

            if not np.all(np.isfinite(vector)):
                raise ValueError("Invalid feature")

            features.append(vector)
            labels.append(label)

        except (KeyError, ValueError, TypeError):
            invalid_rows += 1

if not features:
    raise ValueError("No valid validation samples found.")

X = np.asarray(features, dtype=np.float32)
y_true = np.asarray(labels)

class_names = sorted(set(labels))

print("\n===================================")
print("ORIGINAL vs CANDIDATE MODEL")
print("===================================")

print("Validation samples:", len(y_true))
print("Invalid rows:", invalid_rows)
print("Classes:", class_names)


def evaluate_model(model_path, model_name):
    model = joblib.load(model_path)

    predictions = model.predict(X)

    accuracy = accuracy_score(y_true, predictions) * 100

    print("\n===================================")
    print(model_name)
    print("===================================")

    print("Accuracy: {:.2f}%".format(accuracy))

    print("\nCLASSIFICATION REPORT")
    print("-------------------------")

    print(classification_report(
        y_true,
        predictions,
        labels=class_names,
        zero_division=0
    ))

    print("CONFUSION MATRIX")
    print("-------------------------")

    print("Class order:", class_names)

    print(confusion_matrix(
        y_true,
        predictions,
        labels=class_names
    ))

    return accuracy


original_accuracy = evaluate_model(
    ORIGINAL_MODEL,
    "ORIGINAL MODEL"
)

candidate_accuracy = evaluate_model(
    CANDIDATE_MODEL,
    "CANDIDATE MODEL"
)

print("\n===================================")
print("FINAL COMPARISON")
print("===================================")

print("Original accuracy:  {:.2f}%".format(
    original_accuracy
))

print("Candidate accuracy: {:.2f}%".format(
    candidate_accuracy
))

print("Improvement:        {:+.2f} percentage points".format(
    candidate_accuracy - original_accuracy
))

print("\nNeither model was modified.")
