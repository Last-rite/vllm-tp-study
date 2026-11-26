#!/usr/bin/env bash
# Runs ON THE LOGIN NODE. Each step is ONE job; run them one at a time, in order.
# Never loop over steps: wait for each to finish, check its log, then run the next.
#   bash scripts/nano5/steps.sh <step-id>
#   bash scripts/nano5/steps.sh list
set -euo pipefail
S="bash $(dirname "$0")/submit.sh"
export VTP_STEP="${1:-adhoc}"   # step id goes into job name, log name, ledger and .meta.json
case "${1:-list}" in
  # ---- phase 0: checks (cheap) -------------------------------------------
  N0)  $S 2 2 info ;;                                   # driver, GPU, torch/vllm, nsys   ~2 GPU-min max
  N1)  $S 1 6 bench 1 smoke --batches 1,64 --reps 1 ;;   # 1st start: builds compile cache ~6
  N2)  $S 2 6 bench 2 smoke --batches 1,64 --reps 1 ;;   # TP=2 start + cache          ~12
  # ---- phase 1: communication --------------------------------------------
  N3)  $S 2 4 allreduce 2 ;;                            # all-reduce 1 KiB..256 MiB    ~8
  # ---- phase 2: main sweep (TP=1 vs TP=2), two independent runs each ----
  N4)  $S 1 6 bench 1 main ;;                           #                              ~6
  N5)  $S 2 6 bench 2 main ;;                           #                              ~12
  N6)  $S 1 6 bench 1 main ;;                           # second run (cross-job check) ~6
  N7)  $S 2 6 bench 2 main ;;                           #                              ~12
  # ---- phase 3: ablations at TP=2 ------------------------------------------
  N8)  $S 2 6 bench 2 no_car --disable-custom-all-reduce ;;   #                       ~12
  N9)  $S 2 6 bench 2 eager  --enforce-eager ;;               #                       ~12
  N10) $S 1 6 bench 1 eager  --enforce-eager ;;               # TP=1 reference        ~6
  # ---- phase 4: profiling ----------------------------------------------------
  N11) $S 2 6 profile 2 profile ;;                      #                              ~12
  N12) $S 1 6 profile 1 profile ;;                      #                              ~6
  list) grep -E '^\s+N[0-9]+\)' "$0" ;;
  *) echo "unknown step $1"; exit 1;;
esac
