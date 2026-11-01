
import sys
import csv
import os
import cv2
import mediapipe as mp

from custom_gestures import (
    DATA_FILE,
    ALLOWED_LABELS,
    extract_features
)

SAMPLES_PER_SESSION = 150

if len(sys.argv) != 2:
    print("Usage: python collect_custom_gestures.py PALM")
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

file_exists = os.path.exists(DATA_FILE)
count = 0
recording = False

try:
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists or os.path.getsize(DATA_FILE) == 0:
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

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    if recording and count < SAMPLES_PER_SESSION:
                        features = extract_features(hand)
                        writer.writerow([label] + features)
                        f.flush()
                        count += 1

                if count >= SAMPLES_PER_SESSION:
                    recording = False

                status = (
                    "COLLECTING" if recording
                    else "Press SPACE to record"
                )

                cv2.putText(
                    frame,
                    "{}: {}/{}".format(label, count, SAMPLES_PER_SESSION),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2
                )
                cv2.putText(
                    frame, "Q = quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2
                )

                cv2.imshow("Personalized Gesture Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord(" ") and count < SAMPLES_PER_SESSION:
                    recording = True

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Saved {} samples for {}.".format(count, label))
print("Dataset:", DATA_FILE)

