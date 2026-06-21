#!/bin/bash
set -e

cd "$HOME/LTX-2"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
