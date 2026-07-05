#!/usr/bin/env python3
"""
Start TrialOn backend + Cloudflare tunnel only (no pip install).

Use after a successful kaggle_backend_runner.py run, or when models are already cached.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent.parent


def _kill_old() -> None:
    subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
    subprocess.run("pkill -f cloudflared || true", shell=True)
    time.sleep(1)


def _ensure_cloudflared() -> Path:
    cf = BACKEND_DIR / "cloudflared"
    if not cf.exists():
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        print(f"Downloading cloudflared from {url} ...")
        urllib.request.urlretrieve(url, str(cf))
        cf.chmod(0o755)
    return cf


def _stream_output(proc: subprocess.Popen, prefix: str) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"{prefix} {line.rstrip()}", flush=True)


def main() -> None:
    os.chdir(BACKEND_DIR)
    sys.path.insert(0, str(BACKEND_DIR))

    env_path = BACKEND_DIR / ".env"
    if not env_path.exists():
        print("ERROR: .env missing — run kaggle_backend_runner.py once first.")
        sys.exit(1)

    model_path = BACKEND_DIR / "models" / "catvton" / "mix-48k-1024" / "attention" / "model.safetensors"
    if not model_path.exists():
        print("ERROR: CatVTON weights missing — run kaggle_backend_runner.py once first.")
        sys.exit(1)

    _kill_old()
    (BACKEND_DIR / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data" / "outputs").mkdir(parents=True, exist_ok=True)

    cf = _ensure_cloudflared()

    print("Starting uvicorn on :8000 ...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_stream_output, args=(backend, "[Backend]"), daemon=True).start()
    time.sleep(4)

    print("Starting Cloudflare tunnel ...")
    tunnel = subprocess.Popen(
        [str(cf), "tunnel", "--url", "http://localhost:8000"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url: str | None = None
    deadline = time.time() + 60
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    while time.time() < deadline and tunnel_url is None:
        if tunnel.stdout is None:
            break
        line = tunnel.stdout.readline()
        if not line:
            if tunnel.poll() is not None:
                break
            time.sleep(0.2)
            continue
        print(f"[Tunnel] {line.rstrip()}", flush=True)
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)

    if tunnel_url:
        print("\n" + "=" * 85)
        print(" TRIALON API ONLINE")
        print("=" * 85)
        print(f" API URL:  {tunnel_url}")
        print(f" Docs:     {tunnel_url}/docs")
        print(f" Health:   {tunnel_url}/health")
        print(f"\n frontend/.env.local → NEXT_PUBLIC_API_URL={tunnel_url}")
        print("=" * 85 + "\n")
    else:
        print("WARNING: Tunnel URL not found in 60s — check [Tunnel] logs above.")

    print("Streaming backend logs (Stop cell to shutdown):\n")
    try:
        while backend.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ...")
        backend.terminate()
        tunnel.terminate()
        backend.wait(timeout=5)
        tunnel.wait(timeout=5)


if __name__ == "__main__":
    main()
