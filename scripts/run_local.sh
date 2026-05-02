#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/local-run"
mkdir -p "$LOG_DIR"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing required command: $1"; exit 1; }
}

echo "[1/6] Checking prerequisites..."
require_cmd python
require_cmd node
require_cmd npm
require_cmd curl
require_cmd java
require_cmd mvn

check_port() {
  local p="$1"
  if lsof -iTCP:"$p" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️ Port $p already in use"
  fi
}

check_port 5001
check_port 8080
check_port 5173

# Optional redis check
if command -v redis-cli >/dev/null 2>&1; then
  if redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis reachable"
  else
    echo "⚠️ Redis not reachable (needed for async job queue)"
  fi
else
  echo "⚠️ redis-cli not found; ensure Redis is running for async jobs"
fi

start_ml() {
  echo "[2/6] Starting ML service & worker..."
  (
    cd "$ROOT_DIR/ml-service"
    python app.py >"$LOG_DIR/ml-service.log" 2>&1
  ) &
  echo $! > "$LOG_DIR/ml-service.pid"
  
  (
    cd "$ROOT_DIR/ml-service"
    rq worker upgrade-render >"$LOG_DIR/ml-worker.log" 2>&1
  ) &
  echo $! > "$LOG_DIR/ml-worker.pid"
}

start_backend() {
  echo "[3/6] Starting backend service..."
  (
    cd "$ROOT_DIR/backend"
    mvn spring-boot:run >"$LOG_DIR/backend.log" 2>&1
  ) &
  echo $! > "$LOG_DIR/backend.pid"
}

start_frontend() {
  echo "[4/6] Starting frontend dev server..."
  (
    cd "$ROOT_DIR/frontend"
    npm run dev -- --host 0.0.0.0 >"$LOG_DIR/frontend.log" 2>&1
  ) &
  echo $! > "$LOG_DIR/frontend.pid"
}

cleanup() {
  echo "Stopping services..."
  for pid_file in "$LOG_DIR"/*.pid; do
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file")
      echo "Killing PID $pid ($(basename "$pid_file" .pid))"
      kill -9 "$pid" 2>/dev/null || true
      rm "$pid_file"
    fi
  done
}

trap cleanup EXIT INT TERM

start_ml
start_backend
start_frontend

echo "[5/6] Waiting for services to initialize..."
sleep 15

echo "[6/6] All services started. Streaming logs (Ctrl+C to stop)..."
tail -f "$LOG_DIR"/*.log
