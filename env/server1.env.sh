# Source on server1:  source env/server1.env.sh
export PROJ="${PROJ:-$HOME/vllm-tp-study}"
export STORE="${STORE:-/tmp2/$USER/vtp-store}"   # big local disk; clean up when done
export VENV="$STORE/venv"
export HF_HOME="$STORE/hf"
export VLLM_CACHE_ROOT="$STORE/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$STORE/cache/inductor"
export TRITON_CACHE_DIR="$STORE/cache/triton"
export XDG_CACHE_HOME="$STORE/cache/xdg"
export PIP_CACHE_DIR="$STORE/cache/pip"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
