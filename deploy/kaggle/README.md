# TrialOn — Running Backend on Kaggle GPU (Free GPU guide)

This guide shows you how to run the **TrialOn FastAPI Backend** on Kaggle's free GPU accelerator (T4 or P100) and connect it to your local Next.js frontend.

---

## 🧹 Clean start (use this after a failed run)

If you hit dependency errors, old tunnels, or bad try-on results, **start fresh** with one notebook cell:

```python
# === KAGGLE CLEAN START (paste as a single cell) ===
import shutil, subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/try-on-backend")
URL = "https://github.com/sudipsaud-i-stem/try-on-backend.git"

# Stop old backend + tunnel
subprocess.run("pkill -f 'uvicorn app.main' || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)

# Full wipe (re-downloads models ~1.3 GB — safest clean start)
if REPO.exists():
    shutil.rmtree(REPO)
    print("Removed old repo")

subprocess.check_call(["git", "clone", "--depth", "1", URL, str(REPO)])
%cd /kaggle/working/try-on-backend
!python deploy/kaggle/kaggle_backend_runner.py
```

**Faster option** (keeps downloaded models, only clears try-on data + pulls latest code):

```python
%cd /kaggle/working/try-on-backend
!python deploy/kaggle/clean_start.py
```

**Nuclear option** (same as full wipe above, via script):

```python
%cd /kaggle/working/try-on-backend
!python deploy/kaggle/clean_start.py --full
```

> **Tip:** Use a **new Kaggle notebook session** (Session → Restart session) before clean start if GPU memory looks stuck.

---

## Why run on Kaggle?
- **Free NVIDIA GPU access** (30 hours/week of T4 x2 or P100).
- **Runs the heavy CatVTON model** and full **HUBA streetwear pipeline** (background extraction, matting, facial restoration, and upscaling) in seconds instead of minutes.
- **Zero local installation** needed for the complex ML models or CUDA drivers.

---

## ⚡ Quick Step-by-Step Instructions

### Step 1: Create a Kaggle Notebook
1. Go to [Kaggle](https://www.kaggle.com/) and log in (create a free account if you don't have one).
2. Click **"+ Create"** -> **"New Notebook"** in the top left.
3. In the right sidebar under **Settings**:
   - **Accelerator**: Choose **GPU T4 x2** (recommended) or **GPU P100**.
   - **Internet on**: **Toggle this ON** (This is crucial to allow the notebook to download PyPI libraries, Cloudflare, and Hugging Face weights).

---

### Step 2: Paste and Run Setup Cells
In your Kaggle notebook, create a new cell, paste the following code, and click the **Run** button:

```python
# Choose ONE of the two methods below to clone your private repository:

# METHOD 1 (Secure - Recommended): Use Kaggle User Secrets
# 1. In Kaggle Notebook, click Add-ons (top menu) -> Secrets.
# 2. Add a new secret with Label: "github_pat" and Value: <your_github_personal_access_token>.
# 3. Enable the checkbox for this notebook to access the secret.
# 4. Run the code below:

import os
from kaggle_secrets import UserSecretsClient

if not os.path.exists('/kaggle/working/try-on-backend'):
    print("Cloning private repository...")
    user_secrets = UserSecretsClient()
    pat = user_secrets.get_secret("github_pat")
    # Replace with your actual github username if different:
    username = "sudipsaud-i-stem" 
    !git clone https://{username}:{pat}@github.com/sudipsaud-i-stem/try-on-backend.git /kaggle/working/try-on-backend
else:
    print("Repository already cloned.")


# --- OR ---


# METHOD 2 (Quick): Direct token paste (do not share your notebook publicly!)
# Replace <YOUR_GITHUB_TOKEN> with your token:
# !git clone https://sudipsaud-i-stem:<YOUR_GITHUB_TOKEN>@github.com/sudipsaud-i-stem/try-on-backend.git /kaggle/working/try-on-backend


# 2. Change directory into the backend project
%cd /kaggle/working/try-on-backend

# 3. Run the automated Kaggle GPU backend runner script
!python deploy/kaggle/kaggle_backend_runner.py
```

---

### Step 3: Connect Local Frontend
1. Copy the `.trycloudflare.com` URL printed in the notebook logs.
2. Open your local project folder on your laptop.
3. Open or create the file `frontend/.env.local` and paste:
   ```env
   NEXT_PUBLIC_API_URL=https://xxxx-xxxx-xxxx.trycloudflare.com
   ```
4. Run your local frontend:
   ```powershell
   cd frontend
   npm run dev
   ```
5. Open http://localhost:3000 in your browser. All API requests (including virtual try-on, image matting/background extraction, and face-fixing) will now be processed instantly by the Kaggle GPU!

---

## 🛠️ Troubleshooting

### 1. "Internet connection error" or HF snapshot failing
- Make sure the **Internet** option in the right-side Kaggle notebook panel is toggled **ON**. If you toggled it on after starting, you may need to restart the session/notebook.

### 2. "CUDA not available" error
- Ensure you set the **Accelerator** option to **GPU T4 x2** or **GPU P100** in Kaggle. If you change the accelerator, Kaggle will restart your session on a GPU VM.

### 4. `EncoderDecoderCache` / peft import error
- Pull latest `main` (includes `peft==0.11.1` pin) and run the **Clean start** cell above.
- Do **not** manually `pip install --upgrade transformers` on Kaggle — it breaks CatVTON.

### 5. GFPGAN / Real-ESRGAN warnings
- `functional_tensor` warnings are fixed in latest `main` via `worker/compat.py`. Clean start + re-run.

---
