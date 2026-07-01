# TrialOn — Running Backend on Kaggle GPU (Free GPU guide)

This guide shows you how to run the **TrialOn FastAPI Backend** on Kaggle's free GPU accelerator (T4 or P100) and connect it to your local Next.js frontend.

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
# 1. Clone the repository to the Kaggle working directory
import os
if not os.path.exists('/kaggle/working/try-on-backend'):
    print("Cloning repository...")
    !git clone https://github.com/sudipsaud-i-stem/try-on-backend.git /kaggle/working/try-on-backend
else:
    print("Repository already cloned.")

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

### 3. Tunnel disconnects or times out
- Cloudflare quick tunnels are free and long-running, but Kaggle notebooks will shut down automatically if they are left completely idle for too long. To keep it alive, keep the browser tab containing the Kaggle notebook open.
