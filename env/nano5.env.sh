# Source this file on nano5 (login node AND inside jobs):  source env/nano5.env.sh
# Edit the two paths below if your 100 GB quota lives somewhere other than $HOME.
export PROJ="${PROJ:-$HOME/vllm-tp-study}"      # this repo
export STORE="${STORE:-$HOME/vtp-store}"        # venv + models + caches (counts toward 100 GB)

export VENV="$STORE/venv"
export HF_HOME="$STORE/hf"                       # model weights
export VLLM_CACHE_ROOT="$STORE/cache/vllm"       # torch.compile / CUDA-graph cache (reused across jobs)
export TORCHINDUCTOR_CACHE_DIR="$STORE/cache/inductor"
export TRITON_CACHE_DIR="$STORE/cache/triton"
export XDG_CACHE_HOME="$STORE/cache/xdg"
export PIP_CACHE_DIR="$STORE/cache/pip"          # purge after install: pip cache purge
export TMPDIR="${TMPDIR:-$STORE/tmp}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Slurm account (project code). Change if yours differs.
export SLURM_ACCT="${SLURM_ACCT:-ACD115218}"

# Activate the venv if it exists.
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
