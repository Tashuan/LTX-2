#!/bin/bash
set -euo pipefail

# ============================================================
# LTX-2.3 RunPod Install Script
# Clones the fork, installs deps, downloads models, starts server
# ============================================================

REPO_URL="https://github.com/Tashuan/LTX-2.git"
CLONE_DIR="$HOME/LTX-2"
HF_TOKEN="${HF_TOKEN:-}"

# Use /workspace for models if available (RunPod large volume), else $HOME/models
# The server reads from Path.home() / "models", so we symlink accordingly.
if [ -d "/workspace" ]; then
  MODELS_DIR="/workspace/models"
  mkdir -p "$MODELS_DIR"
  if [ ! -L "$HOME/models" ]; then
    rm -rf "$HOME/models"
    ln -s "$MODELS_DIR" "$HOME/models"
  fi
else
  MODELS_DIR="$HOME/models"
fi

SKIP_MODELS=false
SKIP_CLONE=false
NO_START=false
WITH_DISTILLED=true

for arg in "$@"; do
  case $arg in
    --skip-models)   SKIP_MODELS=true ;;
    --skip-clone)    SKIP_CLONE=true ;;
    --no-start)      NO_START=true ;;
    --no-distilled)  WITH_DISTILLED=false ;;
  esac
done

echo "=== LTX-2.3 Install ==="
echo "Skip models: $SKIP_MODELS | Skip clone: $SKIP_CLONE | No start: $NO_START"

# --------------------------------------------------
# 1. Install uv
# --------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi

# --------------------------------------------------
# 2. Clone fork
# --------------------------------------------------
if [ "$SKIP_CLONE" = false ]; then
  if [ -d "$CLONE_DIR/.git" ]; then
    echo "Repo already exists at $CLONE_DIR, pulling latest…"
    cd "$CLONE_DIR" && git pull
  else
    echo "Cloning fork to $CLONE_DIR…"
    git clone "$REPO_URL" "$CLONE_DIR"
  fi
fi
cd "$CLONE_DIR"

# --------------------------------------------------
# 3. Install Python deps
# --------------------------------------------------
echo "Installing Python dependencies…"
uv sync --frozen --extra xformers
uv pip install fastapi uvicorn python-multipart firebase-admin hf_transfer

# --------------------------------------------------
# 4. Create model directories
# --------------------------------------------------
mkdir -p "$MODELS_DIR/ltx-2.3"
mkdir -p "$MODELS_DIR/gemma-3-12b"
mkdir -p "$MODELS_DIR/ltx-loras/camera"
mkdir -p "$MODELS_DIR/ltx-loras/hdr"

# --------------------------------------------------
# 5. Firebase service account
# --------------------------------------------------
if [ -n "${FIREBASE_SERVICE_ACCOUNT:-}" ]; then
  echo "Writing Firebase service account…"
  echo "$FIREBASE_SERVICE_ACCOUNT" > "$HOME/firebase-service-account.json"
  echo "Firebase service account written."
else
  echo "FIREBASE_SERVICE_ACCOUNT not set — server will use local file serving only."
fi

# --------------------------------------------------
# 6. Download models
# --------------------------------------------------
if [ "$SKIP_MODELS" = false ]; then
  echo "=== Downloading models ==="

  # Activate venv so huggingface-cli is on PATH
  source "$CLONE_DIR/.venv/bin/activate" 2>/dev/null || true

  # HuggingFace CLI (already installed via uv sync, but ensure)
  if ! command -v huggingface-cli &>/dev/null; then
    uv pip install huggingface_hub
  fi
  if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
  fi

  # Main checkpoints — all files live in the single Lightricks/LTX-2.3 repo
  MAIN_REPO="Lightricks/LTX-2.3"
  MAIN_FILES=(
    "ltx-2.3-22b-dev.safetensors"
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
  )

  for filename in "${MAIN_FILES[@]}"; do
    dest="$MODELS_DIR/ltx-2.3/$filename"
    if [ -f "$dest" ]; then
      echo "  ✓ $filename already exists"
    else
      echo "  ↓ Downloading $filename from $MAIN_REPO…"
      huggingface-cli download "$MAIN_REPO" "$filename" --local-dir "$MODELS_DIR/ltx-2.3" || echo "  ⚠ Failed to download $filename — skipping"
    fi
  done

  # Distilled checkpoint (optional)
  if [ "$WITH_DISTILLED" = true ]; then
    dest="$MODELS_DIR/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
    if [ -f "$dest" ]; then
      echo "  ✓ ltx-2.3-22b-distilled-1.1.safetensors already exists"
    else
      echo "  ↓ Downloading distilled checkpoint…"
      huggingface-cli download "$MAIN_REPO" "ltx-2.3-22b-distilled-1.1.safetensors" --local-dir "$MODELS_DIR/ltx-2.3" || echo "  ⚠ Failed to download distilled checkpoint — skipping"
    fi
  fi

  # Gemma 3 12B — check for actual model files, not just non-empty dir (may contain stale .cache)
  if ls "$MODELS_DIR/gemma-3-12b"/*.safetensors 1>/dev/null 2>&1; then
    echo "  ✓ gemma-3-12b already exists"
  else
    echo "  ↓ Downloading Gemma 3 12B…"
    rm -rf "$MODELS_DIR/gemma-3-12b/.cache" 2>/dev/null || true
    huggingface-cli download "google/gemma-3-12b-it-qat-q4_0-unquantized" --local-dir "$MODELS_DIR/gemma-3-12b" || echo "  ⚠ Failed to download Gemma 3 12B — skipping"
  fi

  # IC-LoRA models — each in its own repo with capitalized name
  declare -A IC_LORAS=(
    ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"]="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
    ["ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"]="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control"
    ["ltx-2-19b-ic-lora-detailer.safetensors"]="Lightricks/LTX-2-19b-IC-LoRA-Detailer"
    ["ltx-2-19b-ic-lora-pose-control.safetensors"]="Lightricks/LTX-2-19b-IC-LoRA-Pose-Control"
    ["ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"]="Lightricks/LTX-2.3-22b-IC-LoRA-LipDub"
  )

  for filename in "${!IC_LORAS[@]}"; do
    repo="${IC_LORAS[$filename]}"
    dest="$MODELS_DIR/ltx-loras/$filename"
    if [ -f "$dest" ]; then
      echo "  ✓ $filename already exists"
    else
      echo "  ↓ Downloading IC-LoRA: $filename…"
      huggingface-cli download "$repo" "$filename" --local-dir "$MODELS_DIR/ltx-loras" || echo "  ⚠ Failed to download $filename (may be gated) — skipping"
    fi
  done

  # Camera LoRAs
  CAMERA_MOVES=("dolly-in" "dolly-left" "dolly-out" "dolly-right" "jib-up" "jib-down" "static")
  for move in "${CAMERA_MOVES[@]}"; do
    filename="ltx-2-19b-lora-camera-control-${move}.safetensors"
    dest="$MODELS_DIR/ltx-loras/camera/$filename"
    if [ -f "$dest" ]; then
      echo "  ✓ camera/$filename already exists"
    else
      echo "  ↓ Downloading camera LoRA: $move…"
      # Camera LoRAs are in separate repos per move, capitalized
      capmove=$(echo "$move" | awk -F'-' '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1))substr($i,2)}1' OFS='-')
      repo_name="Lightricks/LTX-2-19b-LoRA-Camera-Control-$capmove"
      huggingface-cli download "$repo_name" "$filename" --local-dir "$MODELS_DIR/ltx-loras/camera" || echo "  ⚠ Failed to download camera/$filename — skipping"
    fi
  done

  echo "=== Model downloads complete ==="
fi

# --------------------------------------------------
# 7. Create output directory
# --------------------------------------------------
# Use /workspace for outputs if available (larger volume), else /data
if [ -d "/workspace" ]; then
  mkdir -p /workspace/outputs
  if [ ! -L "/data/outputs" ]; then
    rm -rf /data/outputs 2>/dev/null || true
    mkdir -p /data
    ln -s /workspace/outputs /data/outputs 2>/dev/null || true
  fi
else
  mkdir -p /data/outputs
fi

# --------------------------------------------------
# 8. Start server
# --------------------------------------------------
if [ "$NO_START" = false ]; then
  echo "=== Starting LTX-2.3 server on :8000 ==="
  cd "$CLONE_DIR"
  source .venv/bin/activate 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
else
  echo "Install complete. Run start_server.sh to start the server."
fi
