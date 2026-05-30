# GPU Process Exporter — Metrics Schema

## Design Goals

- Drop-in compatible with DCGM Prometheus label conventions
- Extends DCGM's GPU-level metrics with per-process and per-user attribution
- Safe to scrape alongside dcgm-exporter without metric name collisions
- All metrics use the `DCGM_FI_` prefix namespace via custom `DCGM_FI_PROC_*` sub-namespace

---

## Standard DCGM Labels (replicated on all metrics)

| Label         | Source                            | Example                    |
|---------------|-----------------------------------|----------------------------|
| `gpu`         | nvidia-smi GPU index              | `0`                        |
| `UUID`        | nvidia-smi GPU UUID               | `GPU-abc123...`            |
| `device`      | `/dev/nvidia<N>`                  | `nvidia0`                  |
| `modelName`   | nvidia-smi GPU name               | `NVIDIA A100-SXM4-80GB`    |
| `Hostname`    | system hostname                   | `worker-node-1`            |
| `DCGM_FI_DRIVER_VERSION` | nvidia-smi driver query | `535.104.12`          |

## Extended Process Labels (added by this exporter)

| Label           | Source                                    | Example         |
|-----------------|-------------------------------------------|-----------------|
| `pid`           | nvidia-smi compute-apps                   | `12345`         |
| `username`      | `/proc/<pid>/status` Uid -> /etc/passwd   | `alice`         |
| `process_name`  | `/proc/<pid>/comm`                        | `python3`       |
| `container_id`  | `/proc/<pid>/cgroup` (docker slice)       | `30562f3da99d`  |
| `cmdline`       | `/proc/<pid>/cmdline` (truncated 128 ch)  | `python train.py --epochs 100` |

---

## Metric Definitions

### Per-Process Memory

```
# HELP DCGM_FI_PROC_FB_USED Per-process framebuffer memory used (in MiB).
# TYPE DCGM_FI_PROC_FB_USED gauge
DCGM_FI_PROC_FB_USED{
  gpu="0", UUID="GPU-xxx", device="nvidia0", modelName="A100", Hostname="host",
  pid="1234", username="alice", process_name="python3", container_id="30562f3da99d", cmdline="python train.py"
} 4096
```

### Per-Process GPU Utilization (sm utilization sampled)

```
# HELP DCGM_FI_PROC_GPU_UTIL Per-process estimated GPU SM utilization (in %).
# TYPE DCGM_FI_PROC_GPU_UTIL gauge
DCGM_FI_PROC_GPU_UTIL{...} 45
```

### Per-Process Running Status (info metric, always 1)

```
# HELP DCGM_FI_PROC_INFO GPU process info. Value is always 1; use labels for attribution.
# TYPE DCGM_FI_PROC_INFO gauge
DCGM_FI_PROC_INFO{...} 1
```

### Per-User Aggregated Memory

```
# HELP DCGM_FI_USER_FB_USED Per-user total framebuffer memory used across all processes (in MiB).
# TYPE DCGM_FI_USER_FB_USED gauge
DCGM_FI_USER_FB_USED{gpu="0", UUID="GPU-xxx", device="nvidia0", modelName="A100", Hostname="host", username="alice"} 8192
```

### Per-User Process Count

```
# HELP DCGM_FI_USER_PROC_COUNT Number of active GPU processes per user per GPU.
# TYPE DCGM_FI_USER_PROC_COUNT gauge
DCGM_FI_USER_PROC_COUNT{gpu="0", UUID="GPU-xxx", device="nvidia0", modelName="A100", Hostname="host", username="alice"} 2
```

### Scrape Health

```
# HELP DCGM_FI_PROC_EXPORTER_SCRAPE_SUCCESS 1 if last nvidia-smi scrape succeeded, 0 otherwise.
# TYPE DCGM_FI_PROC_EXPORTER_SCRAPE_SUCCESS gauge
DCGM_FI_PROC_EXPORTER_SCRAPE_SUCCESS 1
```

---

## Prometheus Scrape Config (to add alongside dcgm-exporter)

```yaml
scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']

  - job_name: 'gpu-process-exporter'
    static_configs:
      - targets: ['gpu-process-exporter:9401']
```

---

## Grafana Query Examples

**Memory by user across all GPUs:**
```promql
sum by (username) (DCGM_FI_USER_FB_USED)
```

**Top processes by memory on GPU 0:**
```promql
topk(10, DCGM_FI_PROC_FB_USED{gpu="0"})
```

**Active GPU processes per user:**
```promql
DCGM_FI_USER_PROC_COUNT
```

**Container GPU memory breakdown:**
```promql
sum by (container_id) (DCGM_FI_PROC_FB_USED)
```
