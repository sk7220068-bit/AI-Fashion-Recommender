import io
import json
from unittest.mock import patch

from PIL import Image

from app import app


def _png_bytes():
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def test_generate_upgrade_requires_idempotency_key():
    client = app.test_client()
    data = {
        "image": (io.BytesIO(_png_bytes()), "x.png"),
        "upgrade_plan": json.dumps({"occasion": "party"}),
    }
    resp = client.post("/generate-upgrade", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
