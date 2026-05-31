#!/usr/bin/env python3
"""
gpu-process-exporter — DCGM-compatible per-process/per-user GPU metrics exporter.

Exposes metrics at :9401/metrics in Prometheus text format.
Label schema mirrors dcgm-exporter so both can be scraped together and
joined in Grafana without extra relabeling.
"""

import functools
import os
import pwd
import re
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
LISTEN_PORT             = int(os.environ.get("PORT", 9401))
SCRAPE_INTERVAL         = float(os.environ.get("SCRAPE_INTERVAL_SEC", 5))
CMDLINE_MAX_LEN         = int(os.environ.get("CMDLINE_MAX_LEN", 128))
# Enable the per-GPU memory divergence check. Defaults to true; produces zero
# log noise at steady state (only logs when proc_sum diverges from GPU total
# by >5% AND >100 MiB). Set to "false" to disable entirely.
ENABLE_DIVERGENCE_CHECK = os.environ.get("ENABLE_DIVERGENCE_CHECK", "true").lower() == "true"
# Path prefix for /proc — override to /host/proc when running in a container
# with the host PID namespace mounted.
PROC_ROOT        = os.environ.get("PROC_ROOT", "/proc")
# In containers, host UIDs are visible via /proc but pwd.getpwuid reads the
# container's own /etc/passwd → wrong or "unknown" usernames.
# Fix: set HOST_ETC=/host/etc and bind-mount host /etc read-only alongside
# PROC_ROOT=/host/proc so UID→name resolution uses the host's passwd database.
HOST_ETC         = os.environ.get("HOST_ETC", "")
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


@functools.lru_cache(maxsize=None)
def username_for_uid(uid: int) -> str:
    """Resolve UID to username; fall back to str(uid).

    Results are cached indefinitely — UIDs→names are static for a running
    exporter, so the passwd file is scanned at most once per unique UID.
    When HOST_ETC is set, the host's passwd is parsed directly so that
    container-namespace UIDs resolve correctly.
    """
    if HOST_ETC:
        try:
            with open(os.path.join(HOST_ETC, "passwd")) as fh:
                for line in fh:
                    parts = line.strip().split(":")
                    if len(parts) >= 3 and int(parts[2]) == uid:
                        return parts[0]
        except (OSError, ValueError):
            pass
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


# Matches a 64-char lowercase hex container ID in a cgroup path segment.
# Handles Docker, containerd/k8s (cri-containerd-<id>.scope), Podman
# (libpod-<id>.scope), and plain /docker/<id> or /containerd/<id> paths.
_CONTAINER_ID_RE = re.compile(r'(?:^|[-/])([0-9a-f]{64})(?:\.scope)?$')


def container_id_for_pid(pid: int) -> str:
    """
    Extract a container ID from /proc/<pid>/cgroup.
    Returns the first 12 chars (matching `docker ps` short ID) or '' if the
    process is not inside a recognised container cgroup.

    Supports cgroup v1 and v2 layouts for Docker, containerd, Podman, and k8s.
    """
    for line in _read_proc_file(pid, "cgroup").splitlines():
        for part in reversed(line.split("/")):
            m = _CONTAINER_ID_RE.search(part)
            if m:
                return m.group(1)[:12]
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


def run_nvidia_smi_pmon() -> dict:
    """
    Run `nvidia-smi pmon -s u -c 1` and return {(gpu_idx, pid): sm_util_pct}.

    `pmon` emits per-process SM (shader multiprocessor / compute) utilization
    as an integer 0-100.  Processes with no SM activity report "-"; those are
    mapped to 0.0.  Returns an empty dict on any failure so callers can safely
    default to 0 without crashing the scrape.

    pmon column order: gpu  pid  type  sm  mem  enc  dec  command
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "pmon", "-s", "u", "-c", "1"],
            capture_output=True, text=True, timeout=30
        )
        sm_map: dict = {}
        for line in result.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # Need at least: gpu  pid  type  sm
            if len(parts) < 4:
                continue
            try:
                gpu_idx = int(parts[0])
                pid     = int(parts[1])
                sm_val  = parts[3]          # "-" when idle or graphics-only
                sm_util = float(sm_val) if sm_val not in ("-", "N/A") else 0.0
                sm_map[(gpu_idx, pid)] = sm_util
            except (ValueError, IndexError):
                continue
        return sm_map
    except FileNotFoundError:
        print("[WARN] nvidia-smi pmon not available — DCGM_FI_PROC_SM_UTIL will be 0")
        return {}
    except subprocess.TimeoutExpired:
        print("[WARN] nvidia-smi pmon timed out")
        return {}
    except Exception as exc:
        print(f"[WARN] nvidia-smi pmon failed: {exc}")
        return {}


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
        # GPU-level framebuffer used from the XML — keyed by str(gpu_idx),
        # same key as labels["gpu"], so _check_divergence comparisons line up.
        self.gpu_fb_used_mib: dict[str, float] = {}


def collect() -> GpuSnapshot:
    snap = GpuSnapshot()
    root = run_nvidia_smi_xml()
    if root is None:
        return snap

    snap.scrape_ok      = True
    snap.driver_version = _text(root, "driver_version")
    sm_map              = run_nvidia_smi_pmon()

    # gpu_idx from enumerate == NVML enumeration order, which pmon also uses.
    # Both nvidia-smi -q -x and pmon are NVML-level; CUDA_VISIBLE_DEVICES does not affect them.
    # UUID would be a more stable join key but pmon does not emit it.
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
        snap.gpu_fb_used_mib[str(gpu_idx)] = _mib_to_float(
            _text(gpu_elem, "fb_memory_usage/used", "0 MiB")
        )

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
                "_sm_util":     sm_map.get((gpu_idx, pid), 0.0),
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


def _proc_numeric_labels(proc: dict) -> dict:
    """Labels for numeric per-process metrics — omit cmdline to avoid cardinality blowup."""
    return {k: v for k, v in proc.items()
            if k != "cmdline" and not k.startswith("_")}


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
        lines.append(f"DCGM_FI_PROC_FB_USED{_labels_str(_proc_numeric_labels(proc))} {proc['_mem_mib']}")
    lines.append("")

    lines.append("# HELP DCGM_FI_PROC_SM_UTIL Per-process SM (shader multiprocessor / compute) utilization (%).")
    lines.append("# TYPE DCGM_FI_PROC_SM_UTIL gauge")
    for proc in snap.processes:
        lines.append(f"DCGM_FI_PROC_SM_UTIL{_labels_str(_proc_numeric_labels(proc))} {proc['_sm_util']}")
    lines.append("")

    lines.append("# HELP DCGM_FI_PROC_INFO GPU process info. Value is always 1; use labels for attribution.")
    lines.append("# TYPE DCGM_FI_PROC_INFO gauge")
    for proc in snap.processes:
        lines.append(f"DCGM_FI_PROC_INFO{_labels_str(proc)} 1")
    lines.append("")

    # --- Per-user aggregated metrics ---
    # Aggregate: (gpu labels) + username -> total mem, process count, sm_util
    user_agg: dict[tuple, dict] = defaultdict(lambda: {"mem": 0.0, "count": 0, "sm_util": 0.0})
    for proc in snap.processes:
        key = (proc["gpu"], proc["UUID"], proc["device"], proc["modelName"],
               proc["Hostname"], proc["DCGM_FI_DRIVER_VERSION"], proc["username"])
        user_agg[key]["mem"]     += proc["_mem_mib"]
        user_agg[key]["count"]   += 1
        user_agg[key]["sm_util"] = min(100.0, user_agg[key]["sm_util"] + proc["_sm_util"])

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

    lines.append("# HELP DCGM_FI_USER_SM_UTIL Per-user SM utilization activity index — sum of per-process SM util capped at 100. Not a true aggregate; use as an activity indicator only.")
    lines.append("# TYPE DCGM_FI_USER_SM_UTIL gauge")
    for key, agg in user_agg.items():
        gpu, uuid, device, model, host, drv, username = key
        lbl = {
            "gpu": gpu, "UUID": uuid, "device": device, "modelName": model,
            "Hostname": host, "DCGM_FI_DRIVER_VERSION": drv, "username": username,
        }
        lines.append(f"DCGM_FI_USER_SM_UTIL{_labels_str(lbl)} {agg['sm_util']}")
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
    Compare per-process memory sum against the GPU-level FB used from the XML.
    Only logs when the two diverge by more than 5% AND more than 100 MiB —
    helps catch stale caches or processes missed in /proc without steady-state noise.

    Uses data already collected in snap.gpu_fb_used_mib (no extra nvidia-smi call).
    """
    proc_sum: dict[str, float] = defaultdict(float)
    for proc in snap.processes:
        proc_sum[proc["gpu"]] += proc["_mem_mib"]

    for gpu_idx, gpu_mib in snap.gpu_fb_used_mib.items():
        our_mib = proc_sum.get(gpu_idx, 0.0)
        diff    = abs(gpu_mib - our_mib)
        pct     = (diff / gpu_mib * 100) if gpu_mib > 0 else 0
        # Only flag if both the absolute diff AND percentage are large.
        # Small absolute diffs are normal CUDA context / driver overhead.
        if pct > 5 and diff > 100:
            print(f"[CHECK] GPU {gpu_idx}: xml={gpu_mib:.0f} MiB  "
                  f"proc_sum={our_mib:.0f} MiB  diff={diff:.0f} MiB ({pct:.1f}%)  *** DIVERGED")


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
            if snap.scrape_ok and ENABLE_DIVERGENCE_CHECK:
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
