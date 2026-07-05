# Kaggle — Cell: UPDATE code + restart server (keeps cached models, ~1 min)

import os
import subprocess
from pathlib import Path

REPO = Path("/kaggle/working/try-on-backend")
os.chdir("/kaggle/working")
subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)

if not REPO.exists():
    raise SystemExit("Repo missing — run notebook_cell.py (first-time setup) first.")

subprocess.check_call(["git", "fetch", "origin", "main"], cwd=REPO)
subprocess.check_call(["git", "reset", "--hard", "origin/main"], cwd=REPO)
print("At commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip())

os.chdir(REPO)
subprocess.check_call(["python", "deploy/kaggle/kaggle_backend_runner.py", "--start-only"], cwd=REPO)
