
import csv
import os
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from custom_gestures import DATA_FILE, MODEL_FILE, ALLOWED_LABELS


def load_dataset():
    features = []
    labels = []

    with open(DATA_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row.get("label", "").upper()

            if label not in ALLOWED_LABELS:
                continue

            try:
                vector = [
                    float(row["f" + str(i)])
                    for i in range(63)
                ]
            except (KeyError, ValueError):
                continue

            if not np.all(np.isfinite(vector)):
                continue

            features.append(vector)
            labels.append(label)

    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def main():
    if not os.path.exists(MODEL_FILE):
        print("Trained model not found.")
        return

    model = joblib.load(MODEL_FILE)
    X, y = load_dataset()

    print("\nDATASET SUMMARY")
    print("-------------------------")
    print("Total samples:", len(y))
    print("Model classes:", list(model.classes_))

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    print("\nCURRENT MODEL - DATASET ACCURACY")
    print("-------------------------")
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    print("\nCLASSIFICATION REPORT")
    print("-------------------------")
    print(classification_report(
        y,
        predictions,
        labels=model.classes_,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print("-------------------------")
    print("Class order:", list(model.classes_))
    print(confusion_matrix(
        y,
        predictions,
        labels=model.classes_
    ))


if __name__ == "__main__":
    main()
