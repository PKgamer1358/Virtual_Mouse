
import csv
import os
import shutil
import tempfile

FILE = "custom_gesture_validation.csv"
BACKUP = "custom_gesture_validation_backup.csv"

# Read existing validation data
with open(FILE, "r", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Find all FIST samples
fist_rows = [r for r in rows if r["label"].upper() == "FIST"]

if len(fist_rows) != 200:
    raise ValueError(
        "Expected exactly 200 FIST samples, found {}. "
        "No changes made.".format(len(fist_rows))
    )

# Keep only the newest 100 FIST samples
new_fist_rows = fist_rows[-100:]

# Keep every non-FIST sample unchanged
other_rows = [
    r for r in rows if r["label"].upper() != "FIST"
]

# Preserve the original order of all other samples
updated_rows = other_rows + new_fist_rows

# Create a backup before modifying anything
shutil.copy2(FILE, BACKUP)

# Write safely to a temporary file
directory = os.path.dirname(os.path.abspath(FILE))

fd, temp_file = tempfile.mkstemp(
    dir=directory,
    suffix=".csv"
)
os.close(fd)

try:
    with open(temp_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    os.replace(temp_file, FILE)

finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print("FIST validation data refreshed successfully.")
print("Old FIST samples: 100")
print("New FIST samples retained: 100")
print("Total validation samples:", len(updated_rows))
print("Backup created:", BACKUP)
