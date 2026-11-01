
import numpy as np

from enhanced_gesture_features import (
    extract_enhanced_features_from_points
)

# Simulated 21-point hand for a structural test.
points = np.random.default_rng(42).random(
    (21, 3)
).astype(np.float32)

points -= points[0]

scale = np.max(np.linalg.norm(points, axis=1))

if scale > 1e-6:
    points /= scale

features = extract_enhanced_features_from_points(points)

print("Enhanced feature test")
print("---------------------")
print("Feature count:", len(features))
print("Expected count: 93")
print("All finite:", np.all(np.isfinite(features)))

if len(features) == 93 and np.all(np.isfinite(features)):
    print("TEST PASSED")
else:
    print("TEST FAILED")
