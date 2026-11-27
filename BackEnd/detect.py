# detect.py
# Loads the trained YOLOv8n jasmine model and performs detection on an image.
# Draws green bounding boxes around jasmine flowers using OpenCV.
# Counts and returns the total number of jasmine detections.
# This module is used directly by app.py (Flask backend).

import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

# ── 1. PATHS ──────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL  = os.path.join(BACKEND_DIR, "runs", "jasmine_v2", "weights", "best.pt")

# ── 2. LOAD MODEL ─────────────────────────────────────────────────────────────
# Model is loaded once here at module level.
# When app.py imports detect.py, the model loads only once at startup —
# not on every request. This keeps the Flask API fast.

if not os.path.exists(BEST_MODEL):
    raise FileNotFoundError(
        f"Trained model not found at:\n{BEST_MODEL}\n"
        "Please complete training before running detection."
    )

print(f"Loading trained model from: {BEST_MODEL}")
model = YOLO(BEST_MODEL)
print("Model loaded successfully.")

# ── 3. DETECTION SETTINGS ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25   # minimum confidence to count a detection
                               # lower = more detections (may include false positives)
                               # higher = fewer detections (may miss some flowers)
CLASS_ID_JASMINE     = 0      # jasmine is class 0 in our dataset
BOX_COLOR            = (0, 255, 0)   # green in BGR (OpenCV uses BGR not RGB)
BOX_THICKNESS        = 2
LABEL_COLOR          = (0, 255, 0)
FONT                 = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE           = 0.6
FONT_THICKNESS       = 2

# ── 4. MAIN DETECTION FUNCTION ────────────────────────────────────────────────
def detect_jasmine(image_path, confidence=CONFIDENCE_THRESHOLD):
    """
    Runs jasmine detection on an image file.

    Args:
        image_path  : full path to the input image file
        confidence  : detection confidence threshold (0.0 to 1.0)

    Returns:
        annotated_image : numpy array (BGR) with green bounding boxes drawn
        jasmine_count   : integer count of detected jasmine flowers
        detections      : list of dicts with box coordinates and confidence
    """

    # ── 4a. Load image ────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at: {image_path}")

    original_image = image.copy()  # keep a clean copy for reference

    # ── 4b. Run YOLO inference ────────────────────────────────────────────────
    # verbose=False suppresses per-image console output during Flask requests
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source     = image_path,
        conf       = confidence,
        device     = device,
        verbose    = False,
        save       = False,   # we handle saving ourselves with OpenCV
    )

    # ── 4c. Parse detections ──────────────────────────────────────────────────
    jasmine_count = 0
    detections    = []

    for result in results:
        boxes = result.boxes  # all detected bounding boxes in this image

        for box in boxes:
            class_id   = int(box.cls[0])      # which class was detected
            conf_score = float(box.conf[0])   # confidence score 0.0 to 1.0

            # Only process jasmine detections (class 0)
            # All other classes are completely ignored
            if class_id != CLASS_ID_JASMINE:
                continue

            # Get bounding box coordinates in pixel values (xyxy format)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Count this jasmine detection
            jasmine_count += 1

            # Store detection info
            detections.append({
                "id"        : jasmine_count,
                "confidence": round(conf_score, 4),
                "box"       : {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

            # ── 4d. Draw green bounding box using OpenCV ──────────────────
            cv2.rectangle(
                image,
                (x1, y1),   # top-left corner
                (x2, y2),   # bottom-right corner
                BOX_COLOR,
                BOX_THICKNESS
            )

            # ── 4e. Draw label with confidence score ──────────────────────
            label = f"Jasmine {conf_score:.2f}"

            # Draw a filled rectangle behind the text for readability
            (text_w, text_h), _ = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )
            cv2.rectangle(
                image,
                (x1, y1 - text_h - 8),
                (x1 + text_w, y1),
                BOX_COLOR,
                -1  # filled rectangle
            )

            # Draw the label text in black over the green background
            cv2.putText(
                image,
                label,
                (x1, y1 - 4),
                FONT,
                FONT_SCALE,
                (0, 0, 0),   # black text
                FONT_THICKNESS
            )

    # ── 4f. Draw total count on the image ─────────────────────────────────
    count_text = f"Jasmine Count: {jasmine_count}"

    # Background rectangle for count text
    (cw, ch), _ = cv2.getTextSize(count_text, FONT, 1.0, 2)
    cv2.rectangle(image, (10, 10), (10 + cw + 10, 10 + ch + 10), (0, 180, 0), -1)

    # Count text
    cv2.putText(
        image,
        count_text,
        (15, 10 + ch),
        FONT,
        1.0,
        (255, 255, 255),  # white text
        2
    )

    return image, jasmine_count, detections

# ── 5. COUNTING ERROR EXPLANATION ─────────────────────────────────────────────
# Common reasons the count may be inaccurate:
#
# Overcounting:
#   - Overlapping flowers may each get their own box (correct behaviour)
#   - High contrast backgrounds may trigger false positives
#   - Lower confidence threshold detects more but risks false positives
#
# Undercounting:
#   - Heavily occluded flowers (hidden behind others) may be missed
#   - Very small flowers far from camera may fall below detection threshold
#   - Flowers at extreme angles may not match learned features well
#
# Adjust the confidence threshold to balance precision vs recall:
#   - confidence=0.25 → more detections, some may be wrong
#   - confidence=0.50 → fewer detections, higher accuracy per detection

# ── 6. QUICK TEST (run this file directly to test detection) ──────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py <path_to_image>")
        print("Example: python detect.py test_flower.jpg")
        sys.exit(1)

    test_image_path = sys.argv[1]

    print(f"Running detection on: {test_image_path}")
    annotated_img, count, dets = detect_jasmine(test_image_path)

    print(f"Jasmine Count: {count}")
    for d in dets:
        print(f"  Flower #{d['id']}: confidence={d['confidence']}, box={d['box']}")

    # Save annotated output
    output_path = "detection_output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"Annotated image saved to: {output_path}")

    # Display the result
    cv2.imshow("Jasmine Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
