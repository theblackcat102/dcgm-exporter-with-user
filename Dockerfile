FROM python:3.11-slim

LABEL org.opencontainers.image.title="gpu-process-exporter"
LABEL org.opencontainers.image.description="DCGM-compatible per-process/per-user GPU metrics exporter"

# nvidia-smi must be present — this image is meant to run with --gpus all
# or with nvidia-container-runtime so that the host driver binaries are
# bind-mounted in. We install only the CLI utilities, not the full driver.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sssd-common \
	curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY exporter.py .

# /proc is bind-mounted from the host at runtime (see docker-compose.yml).
# Set PROC_ROOT so the exporter reads host PIDs, not container PIDs.
ENV PROC_ROOT=/host/proc
ENV PORT=9401
ENV SCRAPE_INTERVAL_SEC=15

EXPOSE 9401

USER nobody

ENTRYPOINT ["python", "-u", "exporter.py"]
