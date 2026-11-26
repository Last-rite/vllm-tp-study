#!/usr/bin/env bash
# Runs ON THE COMPUTE NODE (started by submit.sh through srun). Do not run on the login node.
#   job.sh info
#   job.sh allreduce <ngpus>
#   job.sh bench <tp> <tag> [extra bench_tp.py args...]
#   job.sh profile <tp> <tag> [extra bench_tp.py args...]
set -uo pipefail
cd "$(dirname "$0")/../.."
source env/nano5.env.sh
module load cuda/12.4 >/dev/null 2>&1 || true
export OMP_NUM_THREADS=2          # we only have 4 cores (-c4)
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1           # models are pre-downloaded; never download inside a job
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="${MODEL:-$HF_HOME/models/Qwen3-8B}"
# Log name says which experiment it is:
#   nano5_<step>_<what>_<YYYYmmdd-HHMMSS>_<jobid>.log
#   e.g. nano5_N5_bench-tp2-main_20261126-181226_502579.log
export VTP_STEP="${VTP_STEP:-adhoc}"
case "$1" in
  bench|profile) WHAT="$1-tp${2:-?}-${3:-?}" ;;
  allreduce)     WHAT="allreduce-p${2:-?}" ;;
  *)             WHAT="$1" ;;
esac
export VTP_GIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)"
LOG=results/logs/nano5_${VTP_STEP}_${WHAT}_$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID:-nojob}.log
export VTP_LOG="$LOG"
mkdir -p results/logs
exec > >(tee -a "$LOG") 2>&1
echo "== step ${VTP_STEP}  experiment ${WHAT}  git ${VTP_GIT}"
echo "== job ${SLURM_JOB_ID:-?} on $(hostname) at $(date)  args: $*"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv

# Always use the venv interpreter explicitly: never depend on what PATH resolves to
# on the compute node (module load / srun can change it).
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "FATAL: $PY not found. VENV=$VENV (STORE=$STORE). Is the venv created?"; exit 3
fi
export PATH="$VENV/bin:$PATH"; hash -r
# Metadata check only: does not import torch and does not touch the GPU.
"$PY" -c "import sys; from importlib.metadata import version as v; print('python', sys.executable); print('torch', v('torch'), 'vllm', v('vllm'))" \
  || { echo "FATAL: torch/vllm not installed in $VENV"; exit 3; }
LIMIT=330   # hard stop at 5.5 min, before Slurm kills us at 6 min

rc=0
case "$1" in
  info)
    nvidia-smi; nvidia-smi topo -m
    "$PY" -c "import torch,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__);print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    rc=$?
    command -v nsys && nsys --version || ls -d "${CUDA_HOME:-/nonexistent}"/nsight-systems*/ 2>/dev/null || echo "nsys not found"
    ;;
  allreduce)
    timeout $LIMIT "$PY" harness/allreduce_bench.py --ngpus "$2" --platform h100
    rc=$?
    ;;
  bench)
    tp=$2; tag=$3; shift 3
    timeout $LIMIT "$PY" harness/bench_tp.py --config configs/sweep_main.json \
        --model "$MODEL" --tp "$tp" --platform h100 --tag "$tag" "$@"
    rc=$?
    ;;
  profile)
    tp=$2; tag=$3; shift 3
    NSYS=$(command -v nsys || ls "${CUDA_HOME}"/nsight-systems*/bin/nsys 2>/dev/null | head -1)
    [ -z "$NSYS" ] && { echo "nsys not found"; exit 1; }
    mkdir -p results/nsys
    REP="results/nsys/h100_tp${tp}_${tag}_${SLURM_JOB_ID:-x}"
    # --wait=primary: stop when bench_tp.py exits. Without it nsys keeps waiting for
    # vLLM's re-parented EngineCore/worker processes until `timeout` kills it (rc 124).
    timeout 270 "$NSYS" profile -o "$REP" \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --trace-fork-before-exec=true --cuda-graph-trace=node --force-overwrite=true \
        --wait=primary \
        "$PY" harness/bench_tp.py --config configs/sweep_profile.json \
        --model "$MODEL" --tp "$tp" --platform h100 --tag "$tag" "$@"
    rc=$?
    # export the kernel summary inside the job (no processing on the login node)
    timeout 50 "$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
        --output "$REP" "$REP.nsys-rep" || echo "nsys stats failed; run it later on the laptop"
    ;;
  *) echo "unknown mode $1"; exit 2;;
esac
echo "== exit code $rc at $(date)"
exit $rc
