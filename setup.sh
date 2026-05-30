#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building gpu-process-exporter image..."
docker build -t gpu-process-exporter "$SCRIPT_DIR"

echo "Stopping and removing existing gpu-process-exporter container (if any)..."
docker rm -f gpu-process-exporter 2>/dev/null || true

echo "Starting gpu-process-exporter..."
docker run -d \
  --name gpu-process-exporter \
  --restart unless-stopped \
  --pid host \
  -p 9401:9401 \
  -e PORT=9401 \
  -e SCRAPE_INTERVAL_SEC=5 \
  -e CMDLINE_MAX_LEN=128 \
  -e PROC_ROOT=/host/proc \
  -e HOSTNAME_OVERRIDE="${HOSTNAME}" \
  -v /proc:/host/proc:ro \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=utility \
  -v /etc/nsswitch.conf:/etc/nsswitch.conf:ro \
  -v /var/lib/sss/pipes:/var/lib/sss/pipes:ro \
  -v /etc/passwd:/etc/passwd:ro \
  --runtime nvidia \
  gpu-process-exporter

echo ""
echo "gpu-process-exporter started -> http://localhost:9401/metrics"
