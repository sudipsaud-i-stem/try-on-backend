# Kaggle — copy this entire cell into a new notebook
#
# Prerequisites (Kaggle sidebar):
#   Accelerator: GPU T4 x2 (or P100)
#   Internet: ON

import os
import shutil
import subprocess
from pathlib import Path

# IMPORTANT: reset cwd BEFORE delete/clone. If cwd is inside try-on-backend
# when rmtree runs, git fails with "Unable to read current working directory".
WORKING = Path("/kaggle/working")
REPO = WORKING / "try-on-backend"
URL = "https://github.com/sudipsaud-i-stem/try-on-backend.git"

os.chdir(WORKING)
print("cwd:", os.getcwd())

subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True, cwd=WORKING)
subprocess.run("pkill -f cloudflared || true", shell=True, cwd=WORKING)

if REPO.exists():
    shutil.rmtree(REPO)
    print("Removed old repo")

os.chdir(WORKING)
subprocess.check_call(["git", "clone", "--depth", "1", URL, str(REPO)], cwd=WORKING)
print("Cloned commit:", subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
).strip())

os.chdir(REPO)
subprocess.check_call([os.environ.get("PYTHON", "python3"), "deploy/kaggle/kaggle_backend_runner.py"], cwd=REPO)

# When finished, copy the https://xxxx.trycloudflare.com URL printed above.
# On your PC, set frontend/.env.local:
#   NEXT_PUBLIC_API_URL=https://xxxx.trycloudflare.com
# Then: cd frontend && npm run dev
