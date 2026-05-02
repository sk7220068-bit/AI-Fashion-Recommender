import uuid

import numpy as np
from PIL import Image


class ItemSegmenter:
    """Lightweight segmentation utility that emits per-item binary masks + COCO-style RLE."""

    def __init__(self, artifact_store):
        self.artifact_store = artifact_store

    def segment_items(self, image: Image.Image, detections: list[dict], min_confidence: float = 0.35) -> dict:
        width, height = image.size
        request_id = f"req-{uuid.uuid4().hex[:12]}"

        items = []
        det_index = 1
        for det in detections:
            confidence = float(det.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            bbox = det.get("bounding_box") or det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1, width - 1)); x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1)); y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

            item_id = f"det_{det_index:03d}"
            det_index += 1
            mask_img = Image.fromarray(mask * 255, mode="L")
            mask_key = f"masks/{request_id}/{item_id}.png"
            mask_url = self.artifact_store.upload_image(mask_img, mask_key)

            items.append({
                "item_id": item_id,
                "category": det.get("category", "unknown"),
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "mask_rle": self._encode_rle(mask),
                "mask_png_url": mask_url,
                "style": det.get("style", "unknown"),
                "formality_score": det.get("formality_score", det.get("formalityScore", 0.0)),
            })

