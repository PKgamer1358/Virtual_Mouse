
import csv
import os

files = [
    "custom_gesture_data_93.csv",
    "custom_gesture_validation_93.csv"
]

for filename in files:
    if not os.path.exists(filename):
        print("File not found:", filename)
        continue

    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)

        header = next(reader)
        first_row = next(reader)

        print("\nFile:", filename)
        print("Total columns:", len(header))
        print("Feature count:", len(first_row) - 1)
