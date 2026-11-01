
import os
import csv
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from custom_gestures import MODEL_FILE, ALLOWED_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_FILE = os.path.join(
    BASE_DIR, "custom_gesture_validation.csv"
)

# Check files
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Trained model not found: " + MODEL_FILE)

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError("Validation dataset not found.")

# Load existing model without modifying it.
model = joblib.load(MODEL_FILE)

# Load validation data
features = []
labels = []
invalid_rows = 0

with open(VALIDATION_FILE, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        label = row.get("label", "").upper()

        if label not in ALLOWED_LABELS:
            invalid_rows += 1
            continue

        try:
            vector = [
                float(row["f" + str(i)])
                for i in range(63)
            ]

            if not np.all(np.isfinite(vector)):
                raise ValueError("Invalid numeric values")

            features.append(vector)
            labels.append(label)

        except (KeyError, ValueError, TypeError):
            invalid_rows += 1

if not features:
    raise ValueError("No valid validation samples found.")

X = np.asarray(features, dtype=np.float32)
y_true = np.asarray(labels)

# Predict using the original trained model
y_pred = model.predict(X)

class_names = sorted(ALLOWED_LABELS)

print("\n===================================")
print("GESTURE MODEL VALIDATION RESULTS")
print("===================================")

print("\nTotal valid samples:", len(y_true))
print("Invalid rows:", invalid_rows)
print("Model classes:", list(model.classes_))

print("\nVALIDATION ACCURACY")
print("-------------------------")
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("\nCLASSIFICATION REPORT")
print("-------------------------")
print(classification_report(
    y_true,
    y_pred,
    labels=class_names,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print("-------------------------")
print("Class order:", class_names)
print(confusion_matrix(
    y_true,
    y_pred,
    labels=class_names
))

print("\nINCORRECT PREDICTIONS")
print("-------------------------")

wrong = np.where(y_true != y_pred)[0]

if len(wrong) == 0:
    print("No incorrect predictions.")
else:
    for index in wrong[:30]:
        print(
            "Actual: {:8s} | Predicted: {}".format(
                y_true[index],
                y_pred[index]
            )
        )

    if len(wrong) > 30:
        print("... and", len(wrong) - 30, "more.")

print("\nEvaluation completed.")
print("Original model was not modified.")

