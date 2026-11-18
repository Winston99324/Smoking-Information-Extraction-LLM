set -euo pipefail
 
# --- 1. Install (uncomment if not already installed) -----------------------
# pip install vllm --break-system-packages
 
# --- 2. Config ---------------------------------------------------------------
# Small model for a pure pipeline smoke test (~2-3GB in fp16, loads in seconds)
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
 
# If you hit an attention-backend error, uncomment one of these before launch:
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_ATTENTION_BACKEND=FLASHINFER
 
echo "Launching vLLM OpenAI-compatible server: ${MODEL} on port ${PORT}"
 
vllm serve "${MODEL}" \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port "${PORT}"
