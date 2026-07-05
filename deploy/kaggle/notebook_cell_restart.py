# Kaggle — Cell 2: RESTART server only (~30 sec, no pip install)
# Use when setup already finished but cell stopped / tunnel died.

import os
import subprocess
from pathlib import Path

REPO = Path("/kaggle/working/try-on-backend")
os.chdir(REPO)
subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)
subprocess.check_call(["python", "deploy/kaggle/kaggle_backend_runner.py", "--start-only"], cwd=REPO)
