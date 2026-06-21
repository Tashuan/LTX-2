#!/bin/bash
set -euo pipefail

# ============================================================
# LTX-2.3 RunPod Install Script
# Clones the fork, installs deps, downloads models, starts server
# ============================================================

REPO_URL="https://github.com/Tashuan/LTX-2.git"
CLONE_DIR="$HOME/LTX-2"
MODELS_DIR="$HOME/models"
HF_TOKEN="${HF_TOKEN:-}"

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
uv pip install fastapi uvicorn python-multipart firebase-admin

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

  # HuggingFace CLI
  if ! command -v huggingface-cli &>/dev/null; then
    uv pip install huggingface_hub
  fi
  if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
  fi

  # Main checkpoints
  declare -A MODELS=(
    ["ltx-2.3-22b-dev.safetensors"]="Lightricks/ltx-2.3-22b-dev"
    ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]="Lightricks/ltx-2.3-spatial-upscaler-x2-1.1"
    ["ltx-2.3-22b-distilled-lora-384-1.1.safetensors"]="Lightricks/ltx-2.3-22b-distilled-lora-384-1.1"
    ["ltx-2.3-temporal-upscaler-x2-1.0.safetensors"]="Lightricks/ltx-2.3-temporal-upscaler-x2-1.0"
  )

  for filename in "${!MODELS[@]}"; do
    repo="${MODELS[$filename]}"
    dest="$MODELS_DIR/ltx-2.3/$filename"
    if [ -f "$dest" ]; then
      echo "  ✓ $filename already exists"
    else
      echo "  ↓ Downloading $filename from $repo…"
      huggingface-cli download "$repo" "$filename" --local-dir "$MODELS_DIR/ltx-2.3"
    fi
  done

  # Distilled checkpoint (optional)
  if [ "$WITH_DISTILLED" = true ]; then
    dest="$MODELS_DIR/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
    if [ -f "$dest" ]; then
      echo "  ✓ ltx-2.3-22b-distilled-1.1.safetensors already exists"
    else
      echo "  ↓ Downloading distilled checkpoint…"
      huggingface-cli download "Lightricks/ltx-2.3-22b-distilled-1.1" "ltx-2.3-22b-distilled-1.1.safetensors" --local-dir "$MODELS_DIR/ltx-2.3"
    fi
  fi

  # Gemma 3 12B
  if [ -d "$MODELS_DIR/gemma-3-12b" ] && [ "$(ls -A "$MODELS_DIR/gemma-3-12b" 2>/dev/null)" ]; then
    echo "  ✓ gemma-3-12b already exists"
  else
    echo "  ↓ Downloading Gemma 3 12B…"
    huggingface-cli download "google/gemma-3-12b-it-qat-q4_0-unquantized" --local-dir "$MODELS_DIR/gemma-3-12b"
  fi

  # IC-LoRA models
  declare -A IC_LORAS=(
    ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"]="Lightricks/ltx-2.3-22b-ic-lora-union-control-ref0.5"
    ["ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"]="Lightricks/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5"
    ["ltx-2-19b-ic-lora-detailer.safetensors"]="Lightricks/ltx-2-19b-ic-lora-detailer"
    ["ltx-2-19b-ic-lora-pose-control.safetensors"]="Lightricks/ltx-2-19b-ic-lora-pose-control"
    ["ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"]="Lightricks/ltx-2.3-22b-ic-lora-lipdub-0.9"
  )

  for filename in "${!IC_LORAS[@]}"; do
    repo="${IC_LORAS[$filename]}"
    dest="$MODELS_DIR/ltx-loras/$filename"
    if [ -f "$dest" ]; then
      echo "  ✓ $filename already exists"
    else
      echo "  ↓ Downloading IC-LoRA: $filename…"
      huggingface-cli download "$repo" "$filename" --local-dir "$MODELS_DIR/ltx-loras"
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
      huggingface-cli download "Lightricks/ltx-2-19b-lora-camera-control" "$filename" --local-dir "$MODELS_DIR/ltx-loras/camera"
    fi
  done

  echo "=== Model downloads complete ==="
fi

# --------------------------------------------------
# 7. Create output directory
# --------------------------------------------------
mkdir -p /data/outputs

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
