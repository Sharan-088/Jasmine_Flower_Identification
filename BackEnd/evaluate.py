# evaluate.py
# Evaluates the trained YOLOv8n jasmine model on the test dataset.
# Run this after training is complete and best.pt exists.

import torch
import os
from ultralytics import YOLO

# ── 1. PATHS ─────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML   = os.path.join(BACKEND_DIR, "data.yaml")
BEST_MODEL  = os.path.join(BACKEND_DIR, "runs", "jasmine_v2", "weights", "best.pt")

# ── 2. CHECKS ─────────────────────────────────────────────────────────────────
print("=" * 50)
print("EVALUATION")
print("=" * 50)

if not os.path.exists(BEST_MODEL):
    raise FileNotFoundError(
        f"Trained model not found at:\n{BEST_MODEL}\n"
        "Make sure training is complete before running evaluate.py"
    )

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

print(f"Model  : {BEST_MODEL}")
print(f"Data   : {DATA_YAML}")
print(f"Device : {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 50)

# ── 3. LOAD TRAINED MODEL ─────────────────────────────────────────────────────
# This time we load best.pt — the trained weights saved during training.
# This is different from train.py where we loaded yolov8n.yaml.
model = YOLO(BEST_MODEL)

# ── 4. EVALUATE ON TEST SET ───────────────────────────────────────────────────
print("Running evaluation on test set ...")
print("=" * 50)

metrics = model.val(
    data    = DATA_YAML,
    split   = "test",        # evaluate on test split, not val
    device  = 0 if torch.cuda.is_available() else "cpu",
    workers = 0,             # must be 0 on Windows
    verbose = True,
)

# ── 5. DISPLAY RESULTS ────────────────────────────────────────────────────────
print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

precision = metrics.box.p[0]   # precision for class 0 (Jasmine)
recall    = metrics.box.r[0]   # recall for class 0 (Jasmine)
map50     = metrics.box.map50  # mAP at IoU threshold 0.50
map5095   = metrics.box.map    # mAP at IoU thresholds 0.50 to 0.95

print(f"Precision     : {precision:.4f}  — of all jasmine predictions, how many were correct")
print(f"Recall        : {recall:.4f}  — of all real jasmine flowers, how many were found")
print(f"mAP@50        : {map50:.4f}  — main accuracy metric (higher is better, aim for >0.70)")
print(f"mAP@50-95     : {map5095:.4f}  — stricter accuracy across multiple IoU thresholds")
print("=" * 50)

# ── 6. WHAT THE METRICS MEAN ──────────────────────────────────────────────────
print("""
METRIC EXPLANATIONS:
  Precision  — If the model says "this is jasmine", how often is it right?
               0.90 means 90% of detections are genuine jasmine flowers.

  Recall     — Of all actual jasmine flowers in the image, how many did it find?
               0.85 means it found 85% of jasmine flowers, missed 15%.

  mAP@50     — Mean Average Precision at 50% overlap between predicted
               and actual bounding box. The standard YOLO accuracy metric.
               Above 0.70 is good. Above 0.85 is excellent.

  mAP@50-95  — Stricter version. Averages accuracy across IoU thresholds
               from 0.50 to 0.95. Harder to achieve, more comprehensive.
""")

# Save results path
results_dir = os.path.join(BACKEND_DIR, "runs", "jasmine_v2")
print(f"Full results saved to: {results_dir}")
print("Check results.png and confusion_matrix.png inside that folder.")
