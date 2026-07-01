#!/usr/bin/env python3
"""
Kaggle clean start — wipe old runs and boot a fresh backend.

Usage (from an existing clone):
  python deploy/kaggle/clean_start.py          # wipe data + git pull + run
  python deploy/kaggle/clean_start.py --full   # delete repo, re-clone, run

Or paste the notebook cell from deploy/kaggle/README.md into a new Kaggle notebook.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/sudipsaud-i-stem/try-on-backend.git"
WORKING = Path("/kaggle/working")
REPO = WORKING / "try-on-backend"


def run(cmd: str, cwd: Path | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)


def stop_old_services() -> None:
    print("Stopping old uvicorn / cloudflared processes...")
    subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
    subprocess.run("pkill -f cloudflared || true", shell=True)


def wipe_runtime_data(repo: Path) -> None:
    print("Wiping runtime data (uploads, outputs, database)...")
    for folder in ("uploads", "outputs", "db"):
        target = repo / "data" / folder
        if target.exists():
            shutil.rmtree(target)
            print(f"  removed {target}")
    env_file = repo / ".env"
    if env_file.exists():
        env_file.unlink()
        print(f"  removed {env_file}")


def full_reset() -> Path:
    stop_old_services()
    if REPO.exists():
        print(f"Removing {REPO} ...")
        shutil.rmtree(REPO)
    print(f"Cloning {REPO_URL} ...")
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
    return REPO


def soft_reset() -> Path:
    if not REPO.exists():
        print(f"{REPO} not found — doing full clone instead.")
        return full_reset()
    stop_old_services()
    wipe_runtime_data(REPO)
    print("Pulling latest code...")
    run("git fetch origin main && git reset --hard origin/main", cwd=REPO)
    return REPO


def launch_runner(repo: Path) -> None:
    runner = repo / "deploy" / "kaggle" / "kaggle_backend_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")
    print(f"\nLaunching {runner.name} ...\n")
    os_exec = [sys.executable, str(runner)]
    subprocess.check_call(os_exec, cwd=str(repo))


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle clean start for TrialOn backend")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Delete entire repo and re-clone (re-downloads ~1.3 GB CatVTON weights)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" TrialOn — Kaggle clean start")
    print("=" * 60)

    repo = full_reset() if args.full else soft_reset()
    launch_runner(repo)


if __name__ == "__main__":
    main()
