#!/usr/bin/env python3
"""
gpu-process-exporter — DCGM-compatible per-process/per-user GPU metrics exporter.

Exposes metrics at :9401/metrics in Prometheus text format.
Label schema mirrors dcgm-exporter so both can be scraped together and
joined in Grafana without extra relabeling.
"""

import os
import pwd
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock, Thread
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
LISTEN_PORT      = int(os.environ.get("PORT", 9401))
SCRAPE_INTERVAL  = float(os.environ.get("SCRAPE_INTERVAL_SEC", 5))
CMDLINE_MAX_LEN  = int(os.environ.get("CMDLINE_MAX_LEN", 128))
# Path prefix for /proc — override to /host/proc when running in a container
# with the host PID namespace mounted.
PROC_ROOT        = os.environ.get("PROC_ROOT", "/proc")
# Use HOSTNAME_OVERRIDE so the Hostname label matches dcgm-exporter exactly.
# socket.gethostname() inside Docker returns the container ID, not the real host.
HOSTNAME         = os.environ.get("HOSTNAME_OVERRIDE") or socket.gethostname()


# ---------------------------------------------------------------------------
# Helpers: process introspection via /proc
# ---------------------------------------------------------------------------

def _read_proc_file(pid: int, filename: str) -> str:
    """Read a file from /proc/<pid>/<filename>, return '' on any error."""
    try:
        path = os.path.join(PROC_ROOT, str(pid), filename)
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def uid_for_pid(pid: int) -> Optional[int]:
    """Return the real UID of a process by reading /proc/<pid>/status."""
    for line in _read_proc_file(pid, "status").splitlines():
        if line.startswith("Uid:"):
            # Uid: real  effective  saved  filesystem
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
    return None


def username_for_uid(uid: int) -> str:
    """Resolve UID to username; fall back to str(uid)."""
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def process_name_for_pid(pid: int) -> str:
    """Return the comm (short process name) for a PID."""
    comm = _read_proc_file(pid, "comm").strip()
    return comm or "unknown"


def cmdline_for_pid(pid: int, max_len: int = CMDLINE_MAX_LEN) -> str:
    """Return a human-readable command line for a PID (NUL-separated args)."""
    raw = _read_proc_file(pid, "cmdline")
    cmdline = raw.replace("\x00", " ").strip()
    if len(cmdline) > max_len:
        cmdline = cmdline[:max_len] + "..."
    return cmdline or "unknown"


def container_id_for_pid(pid: int) -> str:
    """
    Extract a Docker/containerd container ID from /proc/<pid>/cgroup.
    Returns '' if the process is not inside a container.
    """
    for line in _read_proc_file(pid, "cgroup").splitlines():
        # cgroup v1: 12:devices:/docker/<64-char-id>
        # cgroup v2: 0::/system.slice/docker-<64-char-id>.scope
        parts = line.split("/")
        for part in reversed(parts):
            # Docker ID: exactly 64 hex chars  OR  docker-<64hex>.scope
            candidate = part.replace("docker-", "").replace(".scope", "").strip()
            if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate):
                return candidate[:12]   # short ID to match `docker ps`
    return ""


# ---------------------------------------------------------------------------
# Helpers: nvidia-smi scraping
# ---------------------------------------------------------------------------

def run_nvidia_smi_xml() -> Optional[ET.Element]:
    """
    Run `nvidia-smi -q -x` and return the parsed XML root element.
    Returns None on failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"[WARN] nvidia-smi exited {result.returncode}: {result.stderr.strip()}")
            return None
        return ET.fromstring(result.stdout)
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found — is the NVIDIA driver installed?")
        return None
    except subprocess.TimeoutExpired:
        print("[ERROR] nvidia-smi timed out")
        return None
    except ET.ParseError as exc:
        print(f"[ERROR] Failed to parse nvidia-smi XML: {exc}")
        return None


def _text(element: ET.Element, path: str, default: str = "unknown") -> str:
    node = element.find(path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _mib_to_float(value: str) -> float:
    """Parse '4096 MiB' -> 4096.0, return 0.0 on failure."""
    try:
        return float(value.replace("MiB", "").replace("N/A", "0").strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Core data collection
# ---------------------------------------------------------------------------

class GpuSnapshot:
    """Holds one scrape cycle's worth of GPU + process data."""

    def __init__(self):
        self.hostname        = HOSTNAME
        self.driver_version  = "unknown"
        self.scrape_ok       = False
        # List of dicts, one per GPU
        self.gpus: list[dict] = []
        # List of dicts, one per (gpu, process)
        self.processes: list[dict] = []


def collect() -> GpuSnapshot:
    snap = GpuSnapshot()
    root = run_nvidia_smi_xml()
    if root is None:
        return snap

    snap.scrape_ok      = True
    snap.driver_version = _text(root, "driver_version")

    for gpu_idx, gpu_elem in enumerate(root.findall("gpu")):
        uuid      = _text(gpu_elem, "uuid")
        model     = _text(gpu_elem, "product_name")
        device    = f"nvidia{gpu_idx}"

        # Common label dict reused for every metric on this GPU
        base_labels = {
            "gpu":                    str(gpu_idx),
            "UUID":                   uuid,
            "device":                 device,
            "modelName":              model,
            "Hostname":               snap.hostname,
            "DCGM_FI_DRIVER_VERSION": snap.driver_version,
        }

        snap.gpus.append(base_labels)

        # Per-process entries
        for proc_elem in gpu_elem.findall("processes/process_info"):
            try:
                pid = int(_text(proc_elem, "pid", "0"))
            except ValueError:
                continue

            mem_used_mib = _mib_to_float(_text(proc_elem, "used_memory", "0 MiB"))
            # G = Graphics, C = Compute, M = Mixed
            proc_type    = _text(proc_elem, "type", "unknown")

            # Resolve process identity from /proc
            uid          = uid_for_pid(pid)
            username     = username_for_uid(uid) if uid is not None else "unknown"
            proc_name    = process_name_for_pid(pid)
            cmdline      = cmdline_for_pid(pid)
            container_id = container_id_for_pid(pid)

            snap.processes.append({
                **base_labels,
                "pid":          str(pid),
                "username":     username,
                "process_name": proc_name,
                "process_type": proc_type,
                "container_id": container_id,
                "cmdline":      cmdline,
                "_mem_mib":     mem_used_mib,
            })

    return snap


# ---------------------------------------------------------------------------
# Prometheus text format rendering
# ---------------------------------------------------------------------------

def _labels_str(labels: dict) -> str:
    """Render a label dict as Prometheus label string: {k="v",...}"""
    # Escape backslash, double-quote, and newline in label values
    def escape(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    pairs = ", ".join(f'{k}="{escape(str(v))}"' for k, v in labels.items() if not k.startswith("_"))
    return "{" + pairs + "}"


def render_metrics(snap: GpuSnapshot) -> str:
    lines = []

    def metric(help_text: str, mtype: str, name: str, labels: dict, value: float):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name}{_labels_str(labels)} {value}")

    # --- Scrape health ---
    lines.append("# HELP DCGM_FI_PROC_EXPORTER_SCRAPE_SUCCESS 1 if last nvidia-smi scrape succeeded, 0 otherwise.")
    lines.append("# TYPE DCGM_FI_PROC_EXPORTER_SCRAPE_SUCCESS gauge")
    lines.append(f"DCGM_FI_PROC_EXPORTER_SCRAPE_SUCCESS {1 if snap.scrape_ok else 0}")
    lines.append("")

    if not snap.scrape_ok:
        return "\n".join(lines)

    # --- Per-process metrics ---
    lines.append("# HELP DCGM_FI_PROC_FB_USED Per-process framebuffer memory used (in MiB).")
    lines.append("# TYPE DCGM_FI_PROC_FB_USED gauge")
    for proc in snap.processes:
        lines.append(f"DCGM_FI_PROC_FB_USED{_labels_str(proc)} {proc['_mem_mib']}")
    lines.append("")

    lines.append("# HELP DCGM_FI_PROC_INFO GPU process info. Value is always 1; use labels for attribution.")
    lines.append("# TYPE DCGM_FI_PROC_INFO gauge")
    for proc in snap.processes:
        lines.append(f"DCGM_FI_PROC_INFO{_labels_str(proc)} 1")
    lines.append("")

    # --- Per-user aggregated metrics ---
    # Aggregate: (gpu labels) + username -> total mem, process count
    user_agg: dict[tuple, dict] = defaultdict(lambda: {"mem": 0.0, "count": 0})
    for proc in snap.processes:
        key = (proc["gpu"], proc["UUID"], proc["device"], proc["modelName"],
               proc["Hostname"], proc["DCGM_FI_DRIVER_VERSION"], proc["username"])
        user_agg[key]["mem"]   += proc["_mem_mib"]
        user_agg[key]["count"] += 1

    lines.append("# HELP DCGM_FI_USER_FB_USED Per-user total framebuffer memory used across all processes on a GPU (in MiB).")
    lines.append("# TYPE DCGM_FI_USER_FB_USED gauge")
    for key, agg in user_agg.items():
        gpu, uuid, device, model, host, drv, username = key
        lbl = {
            "gpu": gpu, "UUID": uuid, "device": device, "modelName": model,
            "Hostname": host, "DCGM_FI_DRIVER_VERSION": drv, "username": username,
        }
        lines.append(f"DCGM_FI_USER_FB_USED{_labels_str(lbl)} {agg['mem']}")
    lines.append("")

    lines.append("# HELP DCGM_FI_USER_PROC_COUNT Number of active GPU processes per user per GPU.")
    lines.append("# TYPE DCGM_FI_USER_PROC_COUNT gauge")
    for key, agg in user_agg.items():
        gpu, uuid, device, model, host, drv, username = key
        lbl = {
            "gpu": gpu, "UUID": uuid, "device": device, "modelName": model,
            "Hostname": host, "DCGM_FI_DRIVER_VERSION": drv, "username": username,
        }
        lines.append(f"DCGM_FI_USER_PROC_COUNT{_labels_str(lbl)} {agg['count']}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Background scrape loop
# ---------------------------------------------------------------------------

class MetricsCache:
    def __init__(self):
        self._lock    = Lock()
        self._payload = b"# waiting for first scrape\n"

    def update(self, text: str):
        encoded = text.encode("utf-8")
        with self._lock:
            self._payload = encoded

    def get(self) -> bytes:
        with self._lock:
            return self._payload


_cache = MetricsCache()


def _check_divergence(snap: GpuSnapshot):
    """
    Compare our per-process memory sum against DCGM's GPU-level total.
    Logs a warning if they diverge by more than 5% — helps catch stale caches
    or processes we failed to read from /proc.
    """
    # Sum process memory per GPU from our snapshot
    proc_sum: dict[str, float] = defaultdict(float)
    for proc in snap.processes:
        proc_sum[proc["gpu"]] += proc["_mem_mib"]

    # Fetch DCGM FB_USED for comparison
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) != 2:
                continue
            gpu_idx  = parts[0].strip()
            dcgm_mib = float(parts[1].strip())
            our_mib  = proc_sum.get(gpu_idx, 0.0)
            diff     = abs(dcgm_mib - our_mib)
            pct      = (diff / dcgm_mib * 100) if dcgm_mib > 0 else 0
            # Only flag if both the absolute diff AND percentage are large.
            # Small absolute diffs on idle GPUs are normal driver/CUDA context overhead.
            flag     = " *** DIVERGED" if (pct > 5 and diff > 100) else ""
            print(f"[CHECK] GPU {gpu_idx}: dcgm={dcgm_mib:.0f} MiB  "
                  f"proc_sum={our_mib:.0f} MiB  diff={diff:.0f} MiB ({pct:.1f}%){flag}")
    except Exception as exc:
        print(f"[CHECK] divergence check failed: {exc}")


def scrape_loop():
    while True:
        start = time.monotonic()
        try:
            snap    = collect()
            payload = render_metrics(snap)
            _cache.update(payload)
            status  = "ok" if snap.scrape_ok else "FAILED"
            elapsed = time.monotonic() - start
            print(f"[INFO] scrape {status} in {elapsed:.2f}s — "
                  f"{len(snap.gpus)} GPU(s), {len(snap.processes)} process(es)")
            if snap.scrape_ok:
                _check_divergence(snap)
        except Exception as exc:
            print(f"[ERROR] scrape loop exception: {exc}")
        time.sleep(SCRAPE_INTERVAL)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/"):
            body = _cache.get()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress default access log spam; only log errors
        if args and str(args[1]) != "200":
            super().log_message(fmt, *args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[INFO] gpu-process-exporter starting on :{LISTEN_PORT}")
    print(f"[INFO] scrape interval: {SCRAPE_INTERVAL}s")
    print(f"[INFO] /proc root: {PROC_ROOT}")

    # Start background scrape thread
    t = Thread(target=scrape_loop, daemon=True)
    t.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] shutting down")
