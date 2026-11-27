# app.py
# Flask backend API for the Jasmine Flower Detection and Counting System.
# Accepts image uploads from the React frontend, runs YOLO detection,
# and returns the jasmine count and annotated image.

import os
import uuid
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from detect import detect_jasmine  # imports our detection module

# ── 1. FLASK APP SETUP ────────────────────────────────────────────────────────
BACKEND_DIR    = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST  = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "dist"))

# Folders for uploaded and processed images
UPLOAD_FOLDER  = os.path.join(BACKEND_DIR, "static", "uploads")
RESULT_FOLDER  = os.path.join(BACKEND_DIR, "static", "results")

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(
    __name__,
    static_folder  = os.path.join(FRONTEND_DIST, "assets"),
    static_url_path= "/assets",
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

@app.after_request
def add_cors_headers(response):
    """Enable CORS so Vite dev server (port 5173) can talk to Flask seamlessly."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# ── 2. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def allowed_file(filename):
    """Check if the uploaded file has an allowed image extension."""
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def image_to_base64(image_path):
    """
    Convert an image file to a base64 encoded string.
    This lets us send the image directly in the JSON response
    instead of serving it as a separate file.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.rsplit(".", 1)[1].lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    return f"data:{mime};base64,{encoded}"

# ── 3. ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main React frontend page."""
    main_html_path = os.path.join(FRONTEND_DIST, "main.html")
    if os.path.exists(main_html_path):
        return send_from_directory(FRONTEND_DIST, "main.html")
    return (
        "<h2>Frontend build not found.</h2>"
        "<p>Please run <code>cd frontend && npm run build</code> or start the Vite dev server with <code>npm run dev</code>.</p>",
        404
    )

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve built JS/CSS assets from frontend/dist/assets."""
    return send_from_directory(
        os.path.join(FRONTEND_DIST, "assets"),
        filename
    )

@app.route("/detect", methods=["POST"])
def detect():
    """
    Main detection endpoint.

    Accepts: multipart/form-data with:
        - image      : image file (jpg, png, webp)
        - confidence : float between 0.1 and 0.9 (optional, default 0.25)

    Returns JSON:
        {
            "success"            : true,
            "jasmine_count"      : 12,
            "detections"         : [...],
            "original_image_url" : "data:image/jpeg;base64,...",
            "processed_image_url": "data:image/jpeg;base64,..."
        }
    """

    # ── 3a. Validate request ──────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error"  : "No image file found in request. Send the file with key 'image'."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error"  : "No file selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error"  : "Invalid file type. Please upload a JPG, PNG or WEBP image."
        }), 400

    # ── 3b. Get confidence threshold from request ─────────────────────────────
    try:
        confidence = float(request.form.get("confidence", 0.25))
        confidence = max(0.1, min(0.9, confidence))  # clamp between 0.1 and 0.9
    except (ValueError, TypeError):
        confidence = 0.25

    # ── 3c. Save uploaded image ───────────────────────────────────────────────
    # Generate a unique filename to avoid conflicts between requests
    ext           = file.filename.rsplit(".", 1)[1].lower()
    unique_name   = f"{uuid.uuid4().hex}"
    upload_path   = os.path.join(UPLOAD_FOLDER, f"{unique_name}.{ext}")
    result_path   = os.path.join(RESULT_FOLDER, f"{unique_name}_result.jpg")

    file.save(upload_path)

    # ── 3d. Run jasmine detection ─────────────────────────────────────────────
    try:
        annotated_image, jasmine_count, detections = detect_jasmine(
            upload_path,
            confidence=confidence
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error"  : f"Detection failed: {str(e)}"
        }), 500

    # ── 3e. Save annotated result image ──────────────────────────────────────
    cv2.imwrite(result_path, annotated_image)

    # ── 3f. Convert both images to base64 for JSON response ──────────────────
    # Sending as base64 means React can display them directly
    # without needing a separate image URL endpoint
    original_b64  = image_to_base64(upload_path)
    processed_b64 = image_to_base64(result_path)

    # ── 3g. Return JSON response ──────────────────────────────────────────────
    return jsonify({
        "success"            : True,
        "jasmine_count"      : jasmine_count,
        "detections"         : detections,
        "original_image_url" : original_b64,
        "processed_image_url": processed_b64,
        "confidence_used"    : confidence,
    })

@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check endpoint.
    Visit http://localhost:5000/health to verify Flask is running.
    """
    return jsonify({
        "status" : "ok",
        "message": "Jasmine Detection API is running"
    })

# ── 4. START SERVER ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Jasmine Detection Flask API")
    print("=" * 50)
    print("API running at : http://localhost:5000")
    print("Health check   : http://localhost:5000/health")
    print("Detection      : POST http://localhost:5000/detect")
    print("=" * 50)
    app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = False,   # set to True only during development
    )
