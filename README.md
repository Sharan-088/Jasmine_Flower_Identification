# Jasmine Flower Detection and Counting System

A deep learning-based web application that detects and counts individual Jasmine flowers in an uploaded image using YOLOv8, trained from scratch on a custom annotated dataset.

---

## Project Overview

- Upload any flower image (single flower, bouquet, or mixed flowers)
- The system detects **only Jasmine flowers** and ignores all other flower types
- Draws green bounding boxes around each detected Jasmine flower
- Displays the total Jasmine flower count

**Example:**
- Input: Image containing 12 jasmine flowers, 5 roses, 3 lilies
- Output: 12 green bounding boxes, label "Jasmine", Count: 12

---

## Technologies Used

| Layer | Technology |
|---|---|
| Deep Learning | YOLOv8 Nano (Ultralytics) |
| Framework | PyTorch |
| Image Processing | OpenCV |
| Backend API | Flask |
| Frontend | React + Vite |
| Dataset | Roboflow (custom annotated) |

---

## Model Details

- **Architecture:** YOLOv8 Nano
- **Training:** From scratch — no pretrained weights used
- **Dataset:** 2,709 images (2,121 train / 399 validation / 189 test)
- **Classes:** 1 (Jasmine only)
- **Hardware:** NVIDIA RTX 3050 Ti Laptop GPU (4GB VRAM)

---

## Project Structure

```
JasmineFlowerIdentification/
│
├── BackEnd/
│   ├── app.py              # Flask API server
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Model evaluation script
│   ├── detect.py           # Detection and counting module
│   ├── data.yaml           # Dataset configuration
│   └── runs/               # Training outputs and model weights
│       └── jasmine_v1/
│           └── weights/
│               └── best.pt # Trained model checkpoint
│
├── dataset/
│   ├── train/              # 2121 training images + labels
│   ├── valid/              # 399 validation images + labels
│   └── test/               # 189 test images + labels
│
├── frontend/
│   ├── main.html           # React entry HTML
│   ├── vite.config.js      # Vite configuration
│   ├── package.json
│   └── src/
│       ├── main.jsx        # React entry point
│       ├── App.jsx         # Main React component
│       └── design.css      # Main stylesheet
│
├── static/
│   └── design.css          # CSS served by Flask
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Setup and Installation

### Prerequisites
- Python 3.11
- Node.js v18+
- NVIDIA GPU with CUDA support (recommended)

### Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

---

## Running the Application

### Step 1 — Train the model (skip if best.pt already exists)

```bash
python BackEnd/train.py
```

### Step 2 — Evaluate the model

```bash
python BackEnd/evaluate.py
```

### Step 3 — Start Flask backend

```bash
python BackEnd/app.py
```

### Step 4 — Start React frontend (new terminal)

```bash
cd frontend
npm run dev
```

### Step 5 — Open in browser

```
http://localhost:5174/main.html
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check if Flask is running |
| POST | `/detect` | Upload image and run detection |

**POST /detect request:**
```
Content-Type: multipart/form-data
Fields:
  image      — image file (JPG, PNG, WEBP)
  confidence — float 0.1 to 0.9 (default: 0.25)
```

**POST /detect response:**
```json
{
  "success": true,
  "jasmine_count": 12,
  "detections": [...],
  "original_image_url": "data:image/jpeg;base64,...",
  "processed_image_url": "data:image/jpeg;base64,..."
}
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Precision | Of all jasmine predictions, how many were correct |
| Recall | Of all real jasmine flowers, how many were found |
| mAP@50 | Main accuracy metric at 50% IoU threshold |
| mAP@50-95 | Stricter accuracy across multiple IoU thresholds |

---

## Dataset

- Source: Roboflow custom dataset
- Annotation format: YOLO (normalized bounding boxes)
- Single class: `jasmine` (class ID: 0)
- Split: 70% train / 20% validation / 10% test

---

## Author

Sharan Raja  
Intern Project — Jasmine Flower Detection Using YOLO
