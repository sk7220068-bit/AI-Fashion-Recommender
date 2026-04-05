"""
AI Fashion Recommender — Python ML Microservice
================================================
Flask-based REST API that wraps:
  - YOLOv8 clothing detection
  - ResNet50 feature extraction

Runs on port 5001. Called by the Java Spring Boot backend.

Usage:
    pip install -r requirements.txt
    python app.py
"""

import io
import logging
import os
from flask import Flask, request, jsonify
from PIL import Image

from detection.clothing_detector import ClothingDetector
from features.feature_extractor import FeatureExtractor

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload

# ── Initialize ML models (loaded once at startup) ─────────────────────────────
logger.info("Loading ML models...")
detector = ClothingDetector()
extractor = FeatureExtractor()
logger.info("ML models loaded successfully.")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "UP", "service": "Fashion AI ML Service"}), 200


@app.route("/detect", methods=["POST"])
def detect_clothing():
    """
    Detect clothing items in an uploaded image using YOLOv8.

    Request: multipart/form-data with field 'image'
    Response:
    {
        "detected_items": [
            {
                "category": "t-shirt",
                "confidence": 0.92,
                "bounding_box": [x1, y1, x2, y2],
                "dominant_color": "white",
                "style": "casual",
                "formality_score": 0.2
            }
        ],
        "image_width": 640,
        "image_height": 480
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read image from uploaded bytes
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        logger.info(f"Running detection on image {image.size} ({len(image_bytes)//1024}KB)")

        # Run YOLO detection
        detected_items = detector.detect(image)

        logger.info(f"Detected {len(detected_items)} clothing items")

        return jsonify({
            "detected_items": detected_items,
            "image_width": image.width,
            "image_height": image.height,
            "item_count": len(detected_items)
        }), 200

    except Exception as e:
        logger.exception(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/extract-features", methods=["POST"])
def extract_features():
    """
    Extract ResNet50 feature vectors from an uploaded clothing image.

    Request: multipart/form-data with field 'image'
    Response:
    {
        "features": {
            "global": [0.12, 0.34, ...],      // 2048-dim overall image vector
            "regions": {                        // Per-detected-item vectors (if available)
                "t-shirt": [0.11, 0.22, ...],
                "jeans":   [0.44, 0.55, ...]
            }
        }
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        logger.info(f"Extracting features from image {image.size}")

        # Extract global feature vector
        global_vector = extractor.extract(image)

        # Run detection to get regions, then extract per-item features
        detected_items = detector.detect(image)
        region_features = {}

        for item in detected_items:
            bbox = item.get("bounding_box")
            category = item.get("category")
            if bbox and category:
                # Crop to the detected region and extract features
                region = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                region_vector = extractor.extract(region)
                region_features[category] = region_vector

        logger.info(f"Extracted features for {len(region_features)} regions")

        return jsonify({
            "features": region_features if region_features else {"global": global_vector},
            "feature_dim": len(global_vector),
            "global_vector": global_vector
        }), 200

    except Exception as e:
        logger.exception(f"Feature extraction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/classify-category", methods=["POST"])
def classify_category():
    """
    Predict fashion categories using the ResNet50 classification head.
    
    Request: multipart/form-data with field 'image'
    Response: { "predictions": [ {"category": "dress", "probability": 0.95, "rank": 1}, ... ] }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        predictions = extractor.classify_category(image, top_k=5)
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions)
        }), 200

    except Exception as e:
        logger.exception(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/find-similar", methods=["POST"])
def find_similar():
    """
    Find visually similar items using FAISS nearest-neighbor search.
    
    Request: multipart/form-data with field 'image', optional 'top_k' (default 10)
    Response: { "matches": [ {"item_id": "...", "category": "...", "score": 0.85, ...}, ... ] }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        top_k = int(request.form.get("top_k", 10))
        
        matches = extractor.find_similar(image, top_k=top_k)
        
        return jsonify({
            "matches": matches,
            "count": len(matches)
        }), 200

    except Exception as e:
        logger.exception(f"Similarity search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns metadata about the currently loaded ML models."""
    return jsonify({
        "detector": detector.model_info(),
        "extractor": extractor.model_info()
    }), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting ML service on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
