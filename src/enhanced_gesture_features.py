
import numpy as np

# MediaPipe fingertip landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]

# Four consecutive landmarks for each finger
FINGER_LANDMARKS = [
    [1, 2, 3, 4],       # Thumb
    [5, 6, 7, 8],       # Index
    [9, 10, 11, 12],    # Middle
    [13, 14, 15, 16],   # Ring
    [17, 18, 19, 20]    # Little
]

FINGER_BASES = [1, 5, 9, 13, 17]


def angle_between(a, b, c):
    """Calculate normalized angle ABC between 0 and 1."""

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.arccos(cosine) / np.pi)


def extract_enhanced_features_from_points(points):
    """
    Generate 93 features:
    63 normalized landmark coordinates
    10 fingertip distances
    15 unique joint angles
    5 finger extension ratios
    """

    points = np.asarray(points, dtype=np.float32)

    if points.shape != (21, 3):
        raise ValueError(
            "Expected exactly 21 landmarks with 3 coordinates."
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("Landmarks contain invalid values.")

    # 1. Original 63 coordinate features
    coordinate_features = points.flatten().tolist()

    # 2. Ten distances between fingertips
    distance_features = []

    for i in range(5):
        for j in range(i + 1, 5):
            distance = np.linalg.norm(
                points[FINGER_TIPS[i]] -
                points[FINGER_TIPS[j]]
            )

            distance_features.append(float(distance))

    # 3. Fifteen unique joint-angle features
    angle_features = []

    wrist = points[0]

    for finger in FINGER_LANDMARKS:
        base, joint1, joint2, tip = finger

        # Angle at the base, using wrist direction
        angle1 = angle_between(
            wrist,
            points[base],
            points[joint1]
        )

        # Angle at the first finger joint
        angle2 = angle_between(
            points[base],
            points[joint1],
            points[joint2]
        )

        # Angle at the second finger joint
        angle3 = angle_between(
            points[joint1],
            points[joint2],
            points[tip]
        )

        angle_features.extend([
            angle1,
            angle2,
            angle3
        ])

    # 4. Five finger extension ratios
    extension_features = []

    for tip, base in zip(FINGER_TIPS, FINGER_BASES):
        tip_distance = np.linalg.norm(
            points[tip] - wrist
        )

        base_distance = np.linalg.norm(
            points[base] - wrist
        )

        ratio = tip_distance / max(base_distance, 1e-6)

        extension_features.append(float(ratio))

    # Combine all features
    result = (
        coordinate_features +
        distance_features +
        angle_features +
        extension_features
    )

    if len(result) != 93:
        raise RuntimeError(
            "Expected 93 features, got {}".format(len(result))
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("Generated features contain invalid values.")

    return result


def extract_enhanced_features(hand_landmarks):
    """
    Extract enhanced features from MediaPipe hand landmarks.
    Uses wrist-relative and scale normalization.
    """

    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    if points.shape != (21, 3):
        raise ValueError("Expected 21 hand landmarks.")

    # Wrist-relative normalization
    points -= points[0]

    # Scale normalization
    scale = np.max(
        np.linalg.norm(points, axis=1)
    )

    if scale > 1e-6:
        points /= scale

    return extract_enhanced_features_from_points(points)
