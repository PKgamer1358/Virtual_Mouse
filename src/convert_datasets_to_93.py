
import csv
import os
import numpy as np

from enhanced_gesture_features import (
    extract_enhanced_features_from_points
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    (
        os.path.join(BASE_DIR, "custom_gesture_data.csv"),
        os.path.join(BASE_DIR, "custom_gesture_data_93.csv"),
        "TRAINING"
    ),
    (
        os.path.join(BASE_DIR, "custom_gesture_validation.csv"),
        os.path.join(BASE_DIR, "custom_gesture_validation_93.csv"),
        "VALIDATION"
    )
]


def convert_dataset(input_file, output_file, dataset_name):
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    converted = 0
    skipped = 0

    with open(input_file, "r", newline="") as source:
        reader = csv.DictReader(source)

        expected_columns = (
            ["label"] + ["f" + str(i) for i in range(63)]
        )

        if reader.fieldnames != expected_columns:
            raise ValueError(
                dataset_name + ": Unexpected input CSV columns."
            )

        with open(output_file, "w", newline="") as destination:
            fieldnames = (
                ["label"] + ["f" + str(i) for i in range(93)]
            )

            writer = csv.DictWriter(
                destination,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for row in reader:
                try:
                    label = row["label"].strip().upper()

                    values = np.asarray(
                        [float(row["f" + str(i)]) for i in range(63)],
                        dtype=np.float32
                    )

                    if not np.all(np.isfinite(values)):
                        raise ValueError("Invalid features")

                    points = values.reshape(21, 3)

                    enhanced = extract_enhanced_features_from_points(
                        points
                    )

                    if len(enhanced) != 93:
                        raise ValueError("Incorrect feature count")

                    output_row = {"label": label}

                    for i, value in enumerate(enhanced):
                        output_row["f" + str(i)] = value

                    writer.writerow(output_row)
                    converted += 1

                except (KeyError, ValueError, TypeError):
                    skipped += 1

    print("\n{} DATASET".format(dataset_name))
    print("-------------------------")
    print("Converted samples:", converted)
    print("Skipped samples:", skipped)
    print("Output:", output_file)

    return converted


if __name__ == "__main__":
    total = 0

    for input_file, output_file, name in DATASETS:
        total += convert_dataset(
            input_file,
            output_file,
            name
        )

    print("\n==============================")
    print("DATASET CONVERSION COMPLETE")
    print("==============================")
    print("Total converted:", total)
    print("Expected feature count: 93")
    print("Original datasets were not modified.")
