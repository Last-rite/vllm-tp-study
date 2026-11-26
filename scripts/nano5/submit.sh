#!/usr/bin/env bash
# Runs ON THE LOGIN NODE. Wraps srun with nano5 course limits and logs GPU-minutes.
#   bash scripts/nano5/submit.sh <ngpus 1|2> <minutes 1-6> <job.sh args...>
# Example:
#   bash scripts/nano5/submit.sh 2 6 bench 2 main
set -euo pipefail
cd "$(dirname "$0")/../.."
source env/nano5.env.sh
G=$1; M=$2; shift 2
[[ "$G" =~ ^[12]$ ]] || { echo "ngpus must be 1 or 2"; exit 1; }
[[ "$M" =~ ^[1-6]$ ]] || { echo "minutes must be 1..6"; exit 1; }
RUNNING=$(squeue -h -u "$USER" -t RUNNING,PENDING | wc -l)
if [ "$RUNNING" -ge 2 ]; then
  echo "You already have $RUNNING jobs running/pending (limit 2). Wait and retry."; exit 1
fi
mkdir -p results/logs
LEDGER=results/logs/su_ledger.csv
STEP="${VTP_STEP:-adhoc}"
[ -f "$LEDGER" ] || echo "submitted_at,user,ngpus,minutes_limit,max_gpu_min,args,step" > "$LEDGER"
head -1 "$LEDGER" | grep -q ',step$' || sed -i '1s/$/,step/' "$LEDGER"
echo "$(date -Iseconds),$USER,$G,$M,$((G*M)),\"$*\",$STEP" >> "$LEDGER"
export VTP_STEP="$STEP"
CMD=(srun -A "$SLURM_ACCT" -p dev -N1 -n1 -c4 --gpus-per-node="$G" -t "00:0$M:00"
     -J "vtp-$STEP" bash scripts/nano5/job.sh "$@")
echo ">> ${CMD[*]}"
"${CMD[@]}" || echo ">> job returned non-zero (see results/logs/)"
echo ">> finished. Remaining wallet:"
wallet "$SLURM_ACCT" 2>/dev/null || true
