#!/usr/bin/env bash
# Full RTX 4090 campaign on server1. Run inside tmux:  tmux new -s vtp
#   bash scripts/server1/run_all.sh            # everything
#   bash scripts/server1/run_all.sh topo|allreduce|main|ablate|profile
# server1 has no 6-minute limit, but the harness still writes CSV incrementally.
set -uo pipefail
cd "$(dirname "$0")/../.."
source env/server1.env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL="${MODEL:-$HF_HOME/models/Qwen3-8B}"
mkdir -p results/logs results/nsys
LOG=results/logs/server1_$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
what=${1:-all}
snap() { nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv >> results/logs/server1_gpu_snapshots.csv; }

bench() {  # bench <tp> <tag> [args]
  local tp=$1 tag=$2; shift 2
  source scripts/server1/pick_gpus.sh "$tp" || { echo "SKIP tp=$tp $tag (GPUs busy)"; return; }
  snap
  python harness/bench_tp.py --config configs/sweep_main.json --model "$MODEL" \
      --tp "$tp" --platform rtx4090 --tag "$tag" --time-budget 1200 "$@"
}

if [[ $what == all || $what == topo ]]; then
  nvidia-smi; nvidia-smi topo -m | tee results/logs/server1_topo.txt
fi
if [[ $what == all || $what == allreduce ]]; then
  for p in 2 4; do
    source scripts/server1/pick_gpus.sh "$p" && snap && \
      python harness/allreduce_bench.py --ngpus "$p" --platform rtx4090
  done
fi
if [[ $what == all || $what == main ]]; then
  for run in 1 2; do for tp in 1 2 4; do bench "$tp" main; done; done
fi
if [[ $what == all || $what == ablate ]]; then
  for tp in 2 4; do
    bench "$tp" no_car --disable-custom-all-reduce
    bench "$tp" eager  --enforce-eager
  done
  bench 1 eager --enforce-eager
fi
if [[ $what == all || $what == profile ]]; then
  for tp in 1 2 4; do
    source scripts/server1/pick_gpus.sh "$tp" || continue
    nsys profile -o "results/nsys/rtx4090_tp${tp}_profile" --trace=cuda,nvtx --sample=none \
      --cpuctxsw=none --trace-fork-before-exec=true --cuda-graph-trace=node --force-overwrite=true \
      python harness/bench_tp.py --config configs/sweep_profile.json --model "$MODEL" \
      --tp "$tp" --platform rtx4090 --tag profile
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      --output "results/nsys/rtx4090_tp${tp}_profile" "results/nsys/rtx4090_tp${tp}_profile.nsys-rep"
  done
fi
echo "== server1 campaign '$what' finished $(date)"
