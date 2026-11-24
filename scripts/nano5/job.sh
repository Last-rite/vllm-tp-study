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
LOG=results/logs/nano5_$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID:-nojob}_$1.log
mkdir -p results/logs
exec > >(tee -a "$LOG") 2>&1
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
    timeout 270 "$NSYS" profile -o "$REP" \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --trace-fork-before-exec=true --cuda-graph-trace=node --force-overwrite=true \
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
