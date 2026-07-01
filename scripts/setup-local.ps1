# TrialOn — Windows local setup (backend + optional frontend)
# Run from repo root:  .\virtual-tryon-backend\scripts\setup-local.ps1

$ErrorActionPreference = "Stop"
$BackendRoot = Split-Path $PSScriptRoot -Parent
$RepoRoot = Split-Path $BackendRoot -Parent
$VenvPython = Join-Path $BackendRoot "venv\Scripts\python.exe"
$VenvPip = Join-Path $BackendRoot "venv\Scripts\pip.exe"

Write-Host "=== TrialOn local setup ===" -ForegroundColor Cyan
Write-Host "Backend: $BackendRoot"

# 1. Python / venv
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' not found. Install Python 3.11 from https://www.python.org/downloads/"
}
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv (Join-Path $BackendRoot "venv")
}

# 2. Fix corrupted numpy metadata if present
$BadNumpy = Join-Path $BackendRoot "venv\Lib\site-packages\~umpy-1.26.4.dist-info"
if (Test-Path $BadNumpy) {
    Write-Host "Removing corrupted numpy metadata..."
    Remove-Item -Recurse -Force $BadNumpy
}

# 3. Install PyTorch with CUDA (skip if already installed)
Write-Host "Installing / verifying PyTorch (CUDA 12.1)..."
& $VenvPip install --upgrade pip
& $VenvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Core + pipeline requirements
Write-Host "Installing backend requirements..."
& $VenvPip install -r (Join-Path $BackendRoot "requirements.txt")
& $VenvPip install -r (Join-Path $BackendRoot "requirements-pipeline.txt")

# 5. Pin ML stack (CatVTON compatibility)
Write-Host "Pinning peft / transformers stack..."
& $VenvPip install --force-reinstall `
    "peft==0.11.1" "transformers==4.40.2" "diffusers==0.27.2" `
    "accelerate==0.30.0" "huggingface-hub==0.23.0" "tokenizers>=0.19,<0.20"
& $VenvPip install "numpy<2.0.0"

# 6. Verify
Write-Host "Verifying ML dependency stack..."
Push-Location $BackendRoot
& $VenvPython -c "from worker.compat import verify_ml_dependency_stack; verify_ml_dependency_stack()"
Pop-Location

# 7. .env
$EnvFile = Join-Path $BackendRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $BackendRoot ".env.example") $EnvFile
    Write-Host "Created .env from .env.example"
}

# 8. Database seed
Write-Host "Seeding database..."
Push-Location $BackendRoot
& $VenvPython scripts\seed_db.py
Pop-Location

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Start backend:"
Write-Host "  cd $BackendRoot"
Write-Host "  .\venv\Scripts\activate"
Write-Host "  uvicorn app.main:app --host 127.0.0.1 --port 8000"
Write-Host ""
Write-Host "Start frontend (new terminal):"
Write-Host "  cd $RepoRoot\frontend"
Write-Host "  npm install"
Write-Host "  npm run dev"
Write-Host ""
Write-Host "Open http://localhost:3000  (API docs: http://127.0.0.1:8000/docs)"
