# Source inside WSL2 Ubuntu on the laptop:  source env/local.env.sh
export PROJ="${PROJ:-$HOME/vllm-tp-study}"
export STORE="${STORE:-$HOME/vtp-store}"
export VENV="$STORE/venv"
export HF_HOME="$STORE/hf"
export VLLM_CACHE_ROOT="$STORE/cache/vllm"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
