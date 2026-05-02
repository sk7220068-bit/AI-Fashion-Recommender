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
import base64
import json
import uuid
from flask import Flask, request, jsonify, g
from redis import Redis
from rq import Queue
from rq.job import Job
from PIL import Image
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest

from detection.clothing_detector import ClothingDetector
from features.feature_extractor import FeatureExtractor
from render.upgrade_renderer import render_upgrade_preview
from segmentation.item_segmenter import ItemSegmenter
from storage.artifact_store import ArtifactStore
from config.security_config import SecurityConfig

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[SecurityConfig.RATE_LIMIT_DEFAULT])
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload

# ── Initialize ML models (loaded once at startup) ─────────────────────────────
logger.info("Loading ML models...")
detector = ClothingDetector()
extractor = FeatureExtractor()
artifact_store = ArtifactStore()
segmenter = ItemSegmenter(artifact_store)
logger.info("ML models loaded successfully.")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
render_queue = Queue("upgrade-render", connection=redis_conn)


def run_render_job(image_b64: str, upgrade_plan: dict, job_id: str):
    job = Job.get_current_job()
    if job:
        job.meta["stage"] = "preprocess"
        job.meta["progress"] = 15
        job.save_meta()

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if job:
        job.meta["stage"] = "rendering"
        job.meta["progress"] = 65
        job.save_meta()

    result = render_upgrade_preview(image, upgrade_plan, artifact_store=artifact_store, job_id=job_id)

    if job:
        job.meta["stage"] = "done"
        job.meta["progress"] = 100
        job.save_meta()

    return {
        "mainImageUrl": result.get("upgraded_image_url"),
        "variants": result.get("alternatives", [])
    }


@app.before_request
def apply_request_security():
    trace_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    g.trace_id = trace_id

    if SecurityConfig.REQUIRE_API_KEY:
        api_key = request.headers.get("X-API-Key", "")
        if not SecurityConfig.API_KEY or api_key != SecurityConfig.API_KEY:
            return jsonify({"error": "Unauthorized", "traceId": trace_id}), 401

    if request.method == "POST" and request.path == "/generate-upgrade":
        if "Idempotency-Key" not in request.headers:
            return jsonify({"error": "Missing Idempotency-Key header", "traceId": trace_id}), 400


@app.after_request
def add_trace_headers(response):
    trace_id = getattr(g, "trace_id", None)
    if trace_id:
        response.headers["X-Request-Id"] = trace_id
    return response


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "UP", "service": "Fashion AI ML Service"}), 200


@app.route("/segment-items", methods=["POST"])
@limiter.limit("20/minute")
def segment_items():
    """Segment clothing items and return per-item masks (RLE + PNG URL)."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        min_confidence = float(request.form.get("min_confidence", 0.35))
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        detections = detector.detect(image)
        segmented = segmenter.segment_items(image, detections, min_confidence=min_confidence)
        return jsonify(segmented), 200
    except Exception as e:
        logger.exception(f"Segmentation error: {e}")
        return jsonify({"error": str(e)}), 500


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


@app.route("/maintenance/cleanup-artifacts", methods=["POST"])
def cleanup_artifacts():
    retention_days = int(request.args.get("retention_days", os.environ.get("ARTIFACT_RETENTION_DAYS", "7")))
    removed = artifact_store.cleanup_local(retention_days=retention_days)
    return jsonify({"status": "ok", "removed": removed, "retentionDays": retention_days}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns metadata about the currently loaded ML models."""
    return jsonify({
        "detector": detector.model_info(),
        "extractor": extractor.model_info()
    }), 200


@app.route("/render-upgrade-preview", methods=["POST"])
def render_upgrade():
    """
    Render a phase-1 visual upgrade preview image from an original image + upgrade plan.
    Request: multipart/form-data with:
      - image: uploaded image
      - upgrade_plan: JSON string
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        upgrade_plan_raw = request.form.get("upgrade_plan", "{}")
        upgrade_plan = json.loads(upgrade_plan_raw)

        preview_job_id = f"preview_{uuid.uuid4().hex[:8]}"
        result = render_upgrade_preview(image, upgrade_plan, artifact_store=artifact_store, job_id=preview_job_id)
        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Render preview error: {e}")
        return jsonify({
            "upgraded_image_url": None,
            "alternatives": [],
            "status": "failed",
            "error": str(e)
        }), 500


@app.route("/generate-upgrade", methods=["POST"])
@limiter.limit("15/minute")
def generate_upgrade():
    """Create asynchronous upgrade render job and return job id."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        upgrade_plan_raw = request.form.get("upgrade_plan", "{}")
        try:
            upgrade_plan = json.loads(upgrade_plan_raw)
        except Exception as ex:
            raise BadRequest(f"Invalid upgrade_plan JSON: {ex}")
        idempotency_key = request.headers.get("Idempotency-Key")
        idem_cache_key = f"idem:generate-upgrade:{idempotency_key}"
        cached_job_id = redis_conn.get(idem_cache_key)
        if cached_job_id:
            return jsonify({"jobId": cached_job_id.decode("utf-8"), "status": "queued", "idempotent": True}), 202
        requested_job_id = upgrade_plan.get("jobId")
        job_id = requested_job_id if requested_job_id else f"job_{uuid.uuid4().hex[:12]}"

        source_bytes = file.read()
        source_key = f"source/{artifact_store.now_path()}/{job_id}.png"
        source_url = artifact_store.upload_bytes(source_bytes, source_key, content_type="image/png")
        upgrade_plan["sourceImageUrl"] = source_url
        image_b64 = base64.b64encode(source_bytes).decode("utf-8")
        job = render_queue.enqueue(
            run_render_job,
            image_b64,
            upgrade_plan,
            job_id,
            job_id=job_id,
            result_ttl=86400,
            failure_ttl=86400
        )
        job.meta["stage"] = "queued"
        job.meta["progress"] = 0
        job.save_meta()

        redis_conn.setex(idem_cache_key, SecurityConfig.IDEMPOTENCY_TTL_SECONDS, job.id)
        return jsonify({"jobId": job.id, "status": "queued", "sourceImageUrl": source_url, "idempotent": False}), 202
    except Exception as e:
        logger.exception(f"Generate upgrade error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return jsonify({"error": "Job not found"}), 404
    status = job.get_status(refresh=True)
    response = {
        "jobId": job.id,
        "status": status,
        "progress": job.meta.get("progress", 0),
        "stage": job.meta.get("stage", "queued"),
        "result": job.result if status == "finished" else {"mainImageUrl": None, "variants": []},
        "error": job.exc_info if status == "failed" else None
    }
    return jsonify(response), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting ML service on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
