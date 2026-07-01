# Kaggle — copy this entire cell into a new notebook

# Prerequisites (Kaggle sidebar):
#   Accelerator: GPU T4 x2 (or P100)
#   Internet: ON

import shutil, subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/try-on-backend")
URL = "https://github.com/sudipsaud-i-stem/try-on-backend.git"

# Stop any previous run
subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)

# Fresh clone (safest when dependencies failed before)
if REPO.exists():
    shutil.rmtree(REPO)
    print("Removed old repo")

subprocess.check_call(["git", "clone", "--depth", "1", URL, str(REPO)])
%cd /kaggle/working/try-on-backend
!python deploy/kaggle/kaggle_backend_runner.py

# When finished, copy the https://xxxx.trycloudflare.com URL printed above.
# On your PC, set frontend/.env.local:
#   NEXT_PUBLIC_API_URL=https://xxxx.trycloudflare.com
# Then: cd frontend && npm run dev
