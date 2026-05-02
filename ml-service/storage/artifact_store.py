import io
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from PIL import Image

try:
    import boto3
except Exception:
    boto3 = None


class ArtifactStore:
    def __init__(self):
        self.backend = os.environ.get("ARTIFACT_STORE_BACKEND", "local").lower()
        self.public_base_url = os.environ.get("ARTIFACT_PUBLIC_BASE_URL", "http://localhost:5001/artifacts").rstrip("/")
        self.local_root = Path(os.environ.get("ARTIFACT_LOCAL_ROOT", "artifacts"))
        self.bucket = os.environ.get("ARTIFACT_BUCKET", "fashionai-artifacts")
        self.region = os.environ.get("AWS_REGION", "us-east-1")
        self.endpoint_url = os.environ.get("S3_ENDPOINT_URL")

        self.s3_client = None
        if self.backend in {"s3", "minio"} and boto3 is not None:
            self.s3_client = boto3.client("s3", region_name=self.region, endpoint_url=self.endpoint_url)

    def now_path(self):
        t = datetime.now(timezone.utc)
        return f"{t:%Y}/{t:%m}/{t:%d}"

    def upload_image(self, image: Image.Image, key: str) -> str:
        buff = io.BytesIO()
        image.save(buff, format="PNG")
        return self.upload_bytes(buff.getvalue(), key, content_type="image/png")

    def upload_bytes(self, payload: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        if self.s3_client:
            self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=payload, ContentType=content_type)
            return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

        out = self.local_root / key
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        return f"{self.public_base_url}/{key}"


    def cleanup_local(self, retention_days: int = 7) -> int:
        if self.s3_client:
