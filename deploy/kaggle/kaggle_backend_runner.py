#!/usr/bin/env python3
"""
TrialOn — Kaggle GPU Backend Runner

This script automates setting up and running the TrialOn FastAPI backend on Kaggle.
It does the following:
1. Verifies GPU/CUDA availability.
2. Checks for internet access (required to download models).
3. Installs requirements.txt and requirements-pipeline.txt.
4. Creates a Kaggle-optimized .env file.
5. Downloads the CatVTON attention weights and auxiliary model weights (GFPGAN, Real-ESRGAN).
6. Runs database migrations/seeding.
7. Downloads Cloudflared and starts a public tunnel.
8. Boots up the FastAPI backend (uvicorn) and streams logs.

Usage in a Kaggle Notebook:
---------------------------
1. Enable GPU accelerator (T4 x2 or P100) and turn on the "Internet" toggle in the sidebar.
2. Run a notebook cell containing:
   !git clone https://github.com/sudipsaud-i-stem/try-on-backend.git /kaggle/working/try-on-backend
   %cd /kaggle/working/try-on-backend
   !python deploy/kaggle/kaggle_backend_runner.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.request
import subprocess
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
# Since this script lives in <repo_root>/deploy/kaggle/
# BACKEND_DIR is the repo root itself
BACKEND_DIR = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(BACKEND_DIR))


def run_cmd(cmd: str, cwd: str | None = None) -> None:
    """Run a shell command and print outputs."""
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def check_gpu() -> bool:
    """Check if CUDA GPU is available in PyTorch."""
    print("=== Step 1: Checking GPU Status ===")
    try:
        import torch
        available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {available}")
        if available:
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Device Count: {torch.cuda.device_count()}")
            return True
        else:
            print("ERROR: CUDA is not available. Please go to Kaggle Settings (right sidebar) -> Accelerator -> select GPU.")
            return False
    except ImportError:
        print("PyTorch is not installed in this environment.")
        return False


def check_internet() -> bool:
    """Verify internet is working."""
    print("\n=== Step 2: Checking Internet Access ===")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        print("Internet connection verified successfully.")
        return True
    except Exception as exc:
        print(f"ERROR: No internet access. Details: {exc}")
        print("Please ensure the 'Internet' toggle is turned ON in the Kaggle settings panel on the right.")
        return False


def install_dependencies() -> None:
    """Install pip requirements."""
    print("\n=== Step 3: Installing Dependencies ===")
    run_cmd(f"{sys.executable} -m pip install --upgrade pip")

    req_file = BACKEND_DIR / "requirements.txt"
    if req_file.exists():
        run_cmd(f"{sys.executable} -m pip install -r requirements.txt", cwd=str(BACKEND_DIR))
        print(
            "\nNOTE: pip may list dependency conflicts with Kaggle pre-installed packages "
            "(numpy, pydantic, jax, etc.). That is expected — CatVTON needs older pins. "
            "Ignore those warnings if the install ends with 'Successfully installed'.\n"
        )
    else:
        print("requirements.txt not found. Skipping.")

    # Install GFPGAN / Real-ESRGAN without replacing Kaggle's pre-built CUDA torch.
    print("Installing pipeline optional packages (GFPGAN, Real-ESRGAN) — keeping Kaggle torch...")
    print("basicsr builds from source on Kaggle — this step can take 5–10 minutes. Please wait...\n")
    run_cmd(
        f"{sys.executable} -m pip install --no-deps "
        f"'gfpgan==1.3.8' 'basicsr==1.4.2' 'realesrgan==0.3.0'",
        cwd=str(BACKEND_DIR),
    )
    run_cmd(
        f"{sys.executable} -m pip install "
        f"'facexlib>=0.3.0' 'filterpy' 'lmdb' 'yapf' 'addict' 'future' 'tb-nightly'",
        cwd=str(BACKEND_DIR),
    )

    # Kaggle pre-installs peft >= 0.13 which imports EncoderDecoderCache from transformers.
    # CatVTON + diffusers 0.27.2 are validated with peft 0.11.1 + transformers 4.40.2.
    # IMPORTANT: use --no-deps so accelerate/peft do not replace Kaggle's CUDA torch with
    # a PyPI cpu torch (breaks torchvision::nms).
    print("Pinning peft/transformers stack for CatVTON (without upgrading torch)...")
    run_cmd(
        f"{sys.executable} -m pip install --no-deps --force-reinstall "
        f"'peft==0.11.1' 'transformers==4.40.2' 'diffusers==0.27.2' "
        f"'accelerate==0.30.0' 'huggingface-hub==0.23.0' 'safetensors==0.4.3'",
        cwd=str(BACKEND_DIR),
    )
    run_cmd(
        f"{sys.executable} -m pip install 'tokenizers>=0.19,<0.20'",
        cwd=str(BACKEND_DIR),
    )

    # basicsr/gfpgan need numpy 1.x on many Linux images.
    print("Forcing numpy 1.26 + scipy rebuild (fixes dtype size changed ABI errors)...")
    run_cmd(f"{sys.executable} -m pip install 'numpy<2.0.0'", cwd=str(BACKEND_DIR))
    run_cmd(
        f"{sys.executable} -m pip install --force-reinstall 'numpy==1.26.4' 'scipy==1.13.0'",
        cwd=str(BACKEND_DIR),
    )

    print("Pinning Pillow (torchvision breaks if Pillow 12.x is mixed with old PIL caches)...")
    run_cmd(f"{sys.executable} -m pip install --force-reinstall 'Pillow==10.3.0'", cwd=str(BACKEND_DIR))

    fix_torchvision_pair()

    print("Verifying ML dependency compatibility...")
    run_cmd(
        f"{sys.executable} -c \"from worker.compat import ensure_torchvision_functional_tensor, verify_ml_dependency_stack, verify_torchvision_cuda_ops; "
        f"ensure_torchvision_functional_tensor(); verify_ml_dependency_stack(); verify_torchvision_cuda_ops()\"",
        cwd=str(BACKEND_DIR),
    )


def fix_torchvision_pair() -> None:
    """
    Reinstall torchvision matched to Kaggle's CUDA torch.

    The peft/accelerate pin step can replace CUDA torch with a PyPI cpu torch (e.g. 2.12.1),
    while torchvision stays on cu128 — causing 'operator torchvision::nms does not exist'.
    """
    print("\n=== Fixing torch + torchvision CUDA pair ===")
    index = "https://download.pytorch.org/whl/cu128"
    if not Path("/kaggle").exists():
        try:
            import torch

            version = torch.__version__
            if "+cu124" in version:
                index = "https://download.pytorch.org/whl/cu124"
            elif "+cu121" in version:
                index = "https://download.pytorch.org/whl/cu121"
            elif "+cu128" not in version:
                print(f"torch={version}; defaulting to cu128 wheel index.")
        except ImportError:
            pass

    print("Removing torch/torchvision/torchaudio (clearing any PyPI cpu torch mismatch)...")
    subprocess.run(
        f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio",
        shell=True,
        cwd=str(BACKEND_DIR),
    )
    print(f"Installing matched CUDA trio from {index} ...")
    run_cmd(
        f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {index}",
        cwd=str(BACKEND_DIR),
    )




def create_env_file() -> None:
    """Create a Kaggle-optimized .env file."""
    print("\n=== Step 4: Configuring Environment File (.env) ===")
    env_path = BACKEND_DIR / ".env"
    
    env_content = """# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# Storage Paths
UPLOAD_DIR=./data/uploads
OUTPUT_DIR=./data/outputs
MODEL_CACHE_DIR=./models

# Rate Limiting (Relaxed for testing)
TRYON_RATE_LIMIT=100
TRYON_RATE_WINDOW_HOURS=1

# ML Model Setup
DEVICE=cuda
TORCH_DTYPE=float16
INFERENCE_STEPS=35
GUIDANCE_SCALE=3.0
OUTPUT_WIDTH=768
OUTPUT_HEIGHT=1024
MASK_BLUR_FACTOR=2
MASK_ERODE_PIXELS=8
CLOTH_TYPE=upper
INFERENCE_SEED=42
COLOR_PRESERVE_STRENGTH=0.30

# HUBA Advanced Pipeline
ENABLE_HUBA_PIPELINE=true
ENABLE_PIPELINE_STAGE0=true
ENABLE_PIPELINE_STAGE2=true
ENABLE_PIPELINE_STAGE4=true
ENABLE_PIPELINE_STAGE5=false
ENABLE_PIPELINE_STAGE6=true
PIPELINE_DEBUG=false
PIPELINE_MIN_SHORT_EDGE=512
PIPELINE_BLUR_THRESHOLD=80
PIPELINE_PARSE_CONFIDENCE=0.45
PIPELINE_PRE_UPSCALE=true
PIPELINE_AUTO_WHITE_BALANCE=false
PIPELINE_BLEND_MODE=garment_only
PIPELINE_NOISE_MATCH_STRENGTH=0.0
PIPELINE_DEBLOCK=false
PIPELINE_UPSCALE_FACTOR=1.0

# Optional heavy models (BiRefNet/GFPGAN add latency; off for Kaggle tunnel <120s)
ENABLE_BIREFNET=false
ENABLE_GFPGAN=false
ENABLE_REALESRGAN=false

# Model Identifiers
CATVTON_MODEL_ID=zhengchong/CatVTON
CATVTON_BASE_MODEL_ID=runwayml/stable-diffusion-inpainting
CATVTON_ATTN_VERSION=mix

ENABLE_PROMETHEUS=false
LOG_LEVEL=INFO
"""
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"Created .env file at {env_path}")


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress feedback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        print(f"File already exists (and looks valid): {dest_path.name}")
        return

    print(f"Downloading {url} -> {dest_path}")
    
    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}% ({count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_hook)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nFailed to download: {e}")
        # Try wget as a fallback
        print("Trying fallback download via wget...")
        run_cmd(f"wget -O '{dest_path}' '{url}'")


def download_all_models() -> None:
    """Download the core CatVTON weights and the optional enhancement models."""
    print("\n=== Step 5: Downloading Model Weights (approx 10-12 GB total) ===")
    
    # 1. Core CatVTON models
    print("Running download_models.py for CatVTON...")
    run_cmd(f"{sys.executable} scripts/download_models.py", cwd=str(BACKEND_DIR))
    
    # 2. GFPGAN weights
    gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    gfpgan_path = BACKEND_DIR / "models" / "gfpgan" / "GFPGANv1.4.pth"
    download_file(gfpgan_url, gfpgan_path)

    # 3. RealESRGAN weights
    realesrgan_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    realesrgan_path = BACKEND_DIR / "models" / "realesrgan" / "RealESRGAN_x2plus.pth"
    download_file(realesrgan_url, realesrgan_path)


def initialize_database() -> None:
    """Seed the SQLite database."""
    print("\n=== Step 6: Initializing & Seeding Database ===")
    run_cmd(f"{sys.executable} scripts/seed_db.py", cwd=str(BACKEND_DIR))


def download_cloudflared() -> Path:
    """Ensure cloudflared is downloaded and ready."""
    print("\n=== Step 7: Downloading Cloudflare Tunnel Binary ===")
    cf_path = BACKEND_DIR / "cloudflared"
    if not cf_path.exists():
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        print(f"Downloading Cloudflared from {url}...")
        urllib.request.urlretrieve(url, str(cf_path))
        cf_path.chmod(0o755)
        print("Cloudflared downloaded.")
    else:
        print("Cloudflared binary already present.")
    return cf_path


def run_services(cf_path: Path) -> None:
    """Start uvicorn backend and Cloudflare tunnel, then stream logs."""
    print("\n=== Step 8: Starting Services & Creating Tunnel ===")
    
    # Ensure folders exist
    (BACKEND_DIR / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data" / "outputs").mkdir(parents=True, exist_ok=True)

    print("Launching FastAPI uvicorn server in background...")
    backend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for backend to boot up
    time.sleep(3)
    
    print("Launching Cloudflare tunnel in background...")
    tunnel_proc = subprocess.Popen(
        [
            str(cf_path), "tunnel", "--url", "http://localhost:8000"
        ],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Search for the trycloudflare URL
    tunnel_url = None
    start_time = time.time()
    
    print("Waiting for public tunnel URL to generate...")
    while True:
        if time.time() - start_time > 45:
            print("WARNING: Timeout waiting for quick tunnel link.")
            break
            
        line = tunnel_proc.stdout.readline()
        if not line:
            break
        
        # Look for Cloudflare's generated url
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group(0)
            break
            
    if tunnel_url:
        print("\n" + "="*85)
        print(" 🎉 SUCCESS! TRIALON API BACKEND IS NOW ONLINE!")
        print("="*85)
        print(f" 🔗 API URL:  {tunnel_url}")
        print(f" 🔗 Docs URL: {tunnel_url}/docs")
        print(f" 🔗 Health:   {tunnel_url}/health")
        print("\n ACTIONS TO CONNECT FRONTEND:")
        print(" 1. Copy the API URL above.")
        print(" 2. In your local frontend folder, open or create '.env.local'.")
        print(f" 3. Paste: NEXT_PUBLIC_API_URL={tunnel_url}")
        print(" 4. Start your frontend: npm run dev")
        print("="*85 + "\n")
    else:
        print("Could not retrieve Cloudflare tunnel URL automatically.")
        print("Please check if the tunnel process exited or has errors.")
        
    print("Streaming live backend logs (Press Stop in Kaggle to exit):\n")
    try:
        while True:
            # Check if backend crashed
            if backend_proc.poll() is not None:
                print("Backend process exited unexpectedly.")
                break
                
            line = backend_proc.stdout.readline()
            if line:
                print(f"[Backend] {line.strip()}")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping services...")
    finally:
        print("Terminating backend & tunnel...")
        backend_proc.terminate()
        tunnel_proc.terminate()
        backend_proc.wait()
        tunnel_proc.wait()
        print("Shutdown complete.")


def main() -> None:
    if not check_gpu():
        # Ask to proceed anyway or abort
        print("Aborting because GPU/CUDA is required for efficient CatVTON processing.")
        return
        
    if not check_internet():
        print("Aborting because internet access is required to download weights and models.")
        return

    try:
        install_dependencies()
        create_env_file()
        download_all_models()
        initialize_database()
        cf_path = download_cloudflared()
        run_services(cf_path)
    except Exception as exc:
        print(f"\nAn error occurred during execution: {exc}")
        raise


if __name__ == "__main__":
    main()
