# train.py
# Trains YOLOv8n from scratch on the Jasmine flower dataset.
# No pretrained weights are loaded - model is initialized from yolov8n.yaml only.

import torch
import os
from ultralytics import YOLO

# ── 1. CUDA / GPU CHECK ──────────────────────────────────────────────────────
print("=" * 50)
print("DEVICE CHECK")
print("=" * 50)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"CUDA available : YES")
    print(f"GPU            : {gpu_name}")
    print(f"VRAM           : {vram:.1f} GB")
    device = "0"  # use GPU 0
else:
    print("CUDA available : NO — training will use CPU (very slow)")
    device = "cpu"

print("=" * 50)

# ── 2. PATHS ─────────────────────────────────────────────────────────────────
# Build absolute path to data.yaml so it works no matter where
# you run this script from.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML   = os.path.join(BACKEND_DIR, "data.yaml")
RUNS_DIR    = os.path.join(BACKEND_DIR, "runs")

print(f"Backend dir    : {BACKEND_DIR}")
print(f"data.yaml path : {DATA_YAML}")
print(f"Runs output    : {RUNS_DIR}")
print("=" * 50)

# Verify data.yaml exists before starting
if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(
        f"data.yaml not found at {DATA_YAML}\n"
        "Make sure data.yaml is inside the BackEnd folder."
    )

# ── 3. MODEL INITIALIZATION ───────────────────────────────────────────────────
# Check if a previous run was interrupted — if last.pt exists, resume from it.
# Otherwise start fresh from yolov8n.yaml with random weights.
LAST_CHECKPOINT = os.path.join(RUNS_DIR, "jasmine_v2", "weights", "last.pt")

if os.path.exists(LAST_CHECKPOINT):
    print(f"Resuming training from checkpoint: {LAST_CHECKPOINT}")
    model = YOLO(LAST_CHECKPOINT)
    RESUME = True
else:
    print("No checkpoint found. Starting fresh from yolov8n.yaml ...")
    model = YOLO("yolov8n.yaml")  # architecture only — no pretrained weights
    RESUME = False

print("Model initialized with random weights. Training from scratch.")
print(f"Total parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print("=" * 50)

# ── 4. TRAINING CONFIGURATION ────────────────────────────────────────────────
# Settings are tuned for RTX 3050 4GB VRAM.
# If you get an out-of-memory error:
#   - Reduce imgsz from 640 to 416
#   - Reduce batch from 16 to 8 or 4
# If training is too slow:
#   - You can increase batch slightly if VRAM allows

print("Starting training ...")
print("=" * 50)

results = model.train(
    data      = DATA_YAML,   # path to data.yaml
    epochs    = 100,         # 100 epochs
    imgsz     = 640,         # image size — reduce to 416 if out-of-memory
    batch     = 16,          # increased from 8 — more stable gradient updates
    device    = device,      # GPU (0) or CPU
    workers   = 0,           # must be 0 on Windows to avoid multiprocessing errors
    project   = RUNS_DIR,    # save results inside BackEnd/runs/
    name      = "jasmine_v2",# new folder so v1 results are preserved
    pretrained= False,       # confirm no pretrained weights
    exist_ok  = True,        # allow re-running without renaming the output folder
    resume    = RESUME,      # resume from last.pt if training was interrupted
    verbose   = True,        # print training progress

    # ── Augmentation — improves recall and generalisation ──────────────────
    # These create variations of training images so the model learns
    # jasmine features under different conditions without needing more images
    hsv_h     = 0.015,       # hue shift — different colour tints
    hsv_s     = 0.7,         # saturation — vivid vs dull lighting
    hsv_v     = 0.4,         # brightness — dark and bright conditions
    degrees   = 15.0,        # rotation up to 15 degrees
    translate = 0.1,         # shift image position slightly
    scale     = 0.5,         # zoom in and out
    fliplr    = 0.5,         # horizontal flip 50% of the time
    flipud    = 0.1,         # vertical flip 10% of the time
    mosaic    = 1.0,         # combine 4 images — helps detect small flowers
    mixup     = 0.1,         # blend two images — improves generalisation
    copy_paste= 0.1,         # copy jasmine instances into other images

    # ── Learning rate ──────────────────────────────────────────────────────
    lr0       = 0.01,        # initial learning rate
    lrf       = 0.01,        # final learning rate fraction
    warmup_epochs = 5,       # gradual warmup for stable early training

    # ── Early stopping ─────────────────────────────────────────────────────
    # Stops training if no improvement after 50 epochs
    # Prevents wasting time if model has already converged
    patience  = 50,
)

# ── 5. RESULTS ────────────────────────────────────────────────────────────────
print("=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
best_model = os.path.join(RUNS_DIR, "jasmine_v2", "weights", "best.pt")
if os.path.exists(best_model):
    print(f"Best model saved at : {best_model}")
else:
    print("best.pt not found — check the runs folder manually.")
print("=" * 50)
