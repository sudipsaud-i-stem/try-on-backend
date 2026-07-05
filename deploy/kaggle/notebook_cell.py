# Kaggle — Cell 1: FIRST TIME setup (GPU T4 x2 + Internet ON)
# Takes ~15 min first run. Run once per new session if models not cached.

import os
import shutil
import subprocess
from pathlib import Path

WORKING = Path("/kaggle/working")
REPO = WORKING / "try-on-backend"
URL = "https://github.com/sudipsaud-i-stem/try-on-backend.git"

os.chdir(WORKING)
subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)

if REPO.exists():
    shutil.rmtree(REPO)

subprocess.check_call(["git", "clone", "--depth", "1", URL, str(REPO)], cwd=WORKING)
print("Cloned:", subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
).strip())

os.chdir(REPO)
subprocess.check_call(["python", "deploy/kaggle/kaggle_backend_runner.py"], cwd=REPO)

# Copy https://xxxx.trycloudflare.com from output → frontend/.env.local
