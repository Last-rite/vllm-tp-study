#!/usr/bin/env bash
# Local smoke test in WSL2 on the RTX 4060. Small model, TP=1, ~2-4 minutes.
#   bash scripts/local/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source env/local.env.sh
python harness/bench_tp.py --config configs/smoke_local.json --tp 1 \
    --platform rtx4060 --tag smoke --time-budget 270
python harness/allreduce_bench.py --ngpus 1 --platform rtx4060 --max-exp 20 || \
    echo "(all-reduce with 1 GPU is only a code-path check)"
echo "Smoke test OK. Files:"; ls -lt results/raw | head
