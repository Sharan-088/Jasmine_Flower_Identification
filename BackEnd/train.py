'''
train.py
Trains YOLOv8n from scratch on the Jasmine flower dataset.
No pretrained weights are loaded - model is initialized from yolov8n.yaml only.
'''
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
# Load YOLOv8n architecture from YAML config ONLY.
# This creates the model with randomly initialized weights.
# No .pt pretrained file is downloaded or loaded.
print("Loading YOLOv8n architecture from yolov8n.yaml ...")
model = YOLO("yolov8n.yaml")  # architecture only — no pretrained weights
print("Model initialized with random weights. Training from scratch.")
print(f"Total parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print("=" * 50)

# ── 4. TRAINING CONFIGURATION ────────────────────────────────────────────────
# Settings are tuned for RTX 3050 4GB VRAM.
# If you get an out-of-memory error:
#   - Reduce imgsz from 640 to 416
#   - Reduce batch from 8 to 4 or 2
# If training is too slow:
#   - You can increase batch slightly if VRAM allows

print("Starting training ...")
print("=" * 50)

results = model.train(
    data      = DATA_YAML,   # path to data.yaml
    epochs    = 50,         # start with 100, increase to 200 if results are good
    imgsz     = 416,         # image size — reduce to 416 if out-of-memory
    batch     = 16,           # batch size — reduce to 4 or 2 if out-of-memory
    device    = device,      # GPU (0) or CPU
    workers   = 0,           # must be 0 on Windows to avoid multiprocessing errors
    project   = RUNS_DIR,    # save results inside BackEnd/runs/
    name      = "jasmine_v1",# subfolder name for this training run
    pretrained= False,       # confirm no pretrained weights
    exist_ok  = True,        # allow re-running without renaming the output folder
    verbose   = True,        # print training progress
)

# ── 5. RESULTS ────────────────────────────────────────────────────────────────
print("=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
best_model = os.path.join(RUNS_DIR, "jasmine_v1", "weights", "best.pt")
if os.path.exists(best_model):
    print(f"Best model saved at : {best_model}")
else:
    print("best.pt not found — check the runs folder manually.")
print("=" * 50)
