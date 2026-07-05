# Kaggle — FIX NumPy + restart server (~1 min)
# Use when try-on fails with "_ARRAY_API not found" or "numpy 2.x"

import os, subprocess
from pathlib import Path

REPO = Path("/kaggle/working/try-on-backend")
os.chdir("/kaggle/working")
subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)

os.chdir(REPO)
subprocess.check_call(["python", "deploy/kaggle/pin_numpy.py"], cwd=REPO)
subprocess.check_call(["python", "deploy/kaggle/kaggle_backend_runner.py", "--start-only"], cwd=REPO)
