# Run index (nano5, H100)

Each nano5 job writes one log in `results/logs/` and, if it measured anything, a CSV and
a `.meta.json` in `results/raw/`. Logs from 2026-11-27 on are named
`nano5_<step>_<experiment>_<time>_<jobid>.log`, and the step, job id, git commit and log
path are also recorded in the `.meta.json`. The table below labels the earlier runs,
whose log names carry only the time, job id and mode.

**Use in the paper:** only rows marked ✅ in the last column. `smoke` runs and failed jobs
are kept for the record (they document the setup problems in SETUP.md) and are not used.

| job id | step | experiment | log | raw output | status | paper |
|---|---|---|---|---|---|---|
| 497678 | N0 | info | nano5_20261124-230343_497678_info.log | – | fail: no torch on compute node | |
| 497980 | N0 | info | nano5_20261125-004250_497980_info.log | – | ok (2×H100, NV18, driver 550.127.08) | setup |
| 498233 | N1 | bench tp1 smoke | nano5_20261125-025804_498233_bench.log | – | fail: no torch (PATH) | |
| 498255 | N1 | bench tp1 smoke | nano5_20261125-030841_498255_bench.log | – | fail: compute-node Python 3.6 | |
| 498680 | N1 | bench tp1 smoke | nano5_20261125-093759_498680_bench.log | – | fail: FlashInfer sampler JIT (nvcc 12.4) | |
| 498704 | N1 | bench tp1 smoke | nano5_20261125-100032_498704_bench.log | h100_Qwen3-8B_tp1_smoke_20261125-100032 | ok (before comm-path fix) | |
| 498712 | N2 | bench tp2 smoke | nano5_20261125-100530_498712_bench.log | – | fail: FlashInfer all-reduce JIT | |
| 498776 | N2 | bench tp2 smoke | nano5_20261125-104737_498776_bench.log | – | fail: FlashInfer all-reduce JIT | |
| 498785 | N2 | bench tp2 smoke | nano5_20261125-105101_498785_bench.log | – | fail: AR+RMSNorm fusion JIT | |
| 498846 | N2 | bench tp2 smoke | nano5_20261125-113737_498846_bench.log | h100_Qwen3-8B_tp2_smoke_20261125-113737 | ok (CUSTOM→PYNCCL) | |
| 498868 | N3 | allreduce p2 (run 1) | nano5_20261125-114801_498868_allreduce.log | allreduce_h100_p2_20261125-114803 | ok | ✅ |
| 498869 | N4 | bench tp1 main (run 1) | nano5_20261125-114810_498869_bench.log | h100_Qwen3-8B_tp1_main_20261125-114810 | ok, 18 rows | ✅ |
| 498871 | N5 | bench tp2 main (run 1) | nano5_20261125-114926_498871_bench.log | h100_Qwen3-8B_tp2_main_20261125-114926 | ok, 18 rows | ✅ |
| 502575 | N3 | allreduce p2 (run 2) | nano5_20261126-181026_502575_allreduce.log | allreduce_h100_p2_20261126-181029 | ok | ✅ |
| 502576 | N6 | bench tp1 main (run 2) | nano5_20261126-181056_502576_bench.log | h100_Qwen3-8B_tp1_main_20261126-181056 | ok, 18 rows | ✅ |
| 502579 | N7 | bench tp2 main (run 2) | nano5_20261126-181226_502579_bench.log | h100_Qwen3-8B_tp2_main_20261126-181226 | ok, 18 rows | ✅ |

Job 502576 is the second run of step N4 and job 502579 the second run of step N5, so
they are listed as N6 and N7 (the cross-job repeat runs). Steps N6 and N7 are done.

GPU time so far: 18.3 GPU-min of actual run time, summed from the log timestamps. The
worst-case ledger (`logs/su_ledger.csv`, which counts each job's time limit) gives 130
GPU-min. The budget is about 480 GPU-min.
