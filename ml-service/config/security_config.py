import os


class SecurityConfig:
    ENV = os.environ.get("APP_ENV", "dev")
    API_KEY = os.environ.get("ML_SERVICE_API_KEY", "")
    REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"
    RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "60 per minute")
    IDEMPOTENCY_TTL_SECONDS = int(os.environ.get("IDEMPOTENCY_TTL_SECONDS", "3600"))
