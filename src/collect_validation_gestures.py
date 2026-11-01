
import sys
import csv
import os
import cv2
import mediapipe as mp

from custom_gestures import ALLOWED_LABELS, extract_features

# Save validation data separately from training data.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_FILE = os.path.join(
    BASE_DIR, "custom_gesture_validation.csv"
)

SAMPLES_PER_SESSION = 100
FRAME_INTERVAL = 5

if len(sys.argv) != 2:
    print("Usage: python collect_validation_gestures.py PALM")
    sys.exit(1)

label = sys.argv[1].upper()

if label not in ALLOWED_LABELS:
    print("Allowed labels:", ", ".join(sorted(ALLOWED_LABELS)))
    sys.exit(1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open webcam.")
    sys.exit(1)

file_exists = os.path.exists(VALIDATION_FILE)
count = 0
recording = False
frame_count = 0

try:
    with open(VALIDATION_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists or os.path.getsize(VALIDATION_FILE) == 0:
            writer.writerow(
                ["label"] + ["f" + str(i) for i in range(63)]
            )

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            while camera.isOpened():
                ok, frame = camera.read()

                if not ok:
                    print("Could not read camera frame.")
                    break

                frame_count += 1
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Collect one sample every five frames.
                    if (
                        recording
                        and count < SAMPLES_PER_SESSION
                        and frame_count % FRAME_INTERVAL == 0
                    ):
                        features = extract_features(hand)
                        writer.writerow([label] + features)
                        f.flush()
                        count += 1

                if count >= SAMPLES_PER_SESSION:
                    recording = False

                status = (
                    "COLLECTING"
                    if recording
                    else "Press SPACE to start"
                )

                cv2.putText(
                    frame,
                    "{}: {}/{}".format(
                        label, count, SAMPLES_PER_SESSION
                    ),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Q = quit",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Gesture Validation Data Collector",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if (
                    key == ord(" ")
                    and count < SAMPLES_PER_SESSION
                ):
                    recording = True

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nVALIDATION COLLECTION COMPLETE")
print("Gesture:", label)
print("Samples collected:", count)
print("Validation file:", VALIDATION_FILE)
