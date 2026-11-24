# vLLM 張量並行擴展性研究：工作計畫

> **When Does Tensor Parallelism Pay Off? A Communication-Aware Study of vLLM Inference on PCIe and NVLink GPUs**

- 繳交期限：**12/5（五）**。內部目標：**11/30（一）完成全文初稿**
- 論文：`../template/`（ACM SIGPLAN / PPoPP 格式，主檔 `main.tex`）
- 逐步操作指令：**[SETUP.md](SETUP.md)**（本機、nano5、server1、分析）

---

## 1. 研究問題（縮小後的版本）

只回答一個核心問題，分成兩部分：

| | 問題 | 對應課程要求 |
|---|---|---|
| **Q1** | vLLM 的張量並行（TP）在 NVLink（H100）與 PCIe（RTX 4090）上擴展效率差多少？差距有多少來自通訊？ | speedup、strong/weak scaling、profiling、ablation |
| **Q2** | 用單卡數據與 all-reduce 微基準校準的「計算 + 記憶體 + 通訊」模型，能多準確預測多卡結果？ | modeling |

**固定條件：** 模型 Qwen3-8B（BF16），輸入 256、輸出 128 個 token，batch 為 1、4、8、16、32、64。所有平台使用同一個 vLLM 版本。

**刪除的項目：** 4060 的數據（只用來開發）、調校類實驗（chunked prefill、prefix caching、量化）、14B 模型。

---

## 2. 工作分級（依人力取捨）

最壞情況只有一個人時，只做 **Must**。

| 等級 | 項目 | 產出（論文中的位置） |
|---|---|---|
| **Must** | nano5：N0–N5（環境、all-reduce、TP=1/2 主實驗） | Fig. 1 all-reduce、Fig. 2 strong scaling |
| **Must** | server1：topo、allreduce、main（TP=1/2/4） | Fig. 1、Fig. 2、Fig. 3 weak scaling |
| **Must** | 模型擬合與驗證（`analysis/model.py`） | Fig. 5、Table 3 |
| **Should** | 消融：N8–N10、server1 ablate | Table 2 |
| **Should** | Nsight 剖析：N11–N12、server1 profile | Fig. 4 |
| **Could** | 第二輪主實驗（N6–N7）作跨作業驗證 | 誤差棒、可信度 |
| **Could** | DP 複本對照的討論 | §5 Discussion |

**server1 若到 11/27 仍無法使用：** 論文改成只用 H100（TP=1 對 2 加上模型），在 Limitations 誠實說明缺少 PCIe 對照。這是降級方案，不是失敗。

---

## 3. 分工

| | A：實驗 | B：分析與寫作 |
|---|---|---|
| 主要工作 | 本機環境、nano5、server1、harness 除錯、資料回收 | 模型、畫圖、Nsight 分類、論文主筆 |
| 論文章節 | §3 Methodology 的 harness 部分、§4.1 Setup | §1、§2、§3 其餘、§4.2 之後、§5、§6 |

**只有一個人時：** 依照第 5 節的排程依序做。實驗跑的時候（尤其 server1 的長時間實驗）同時寫論文。
**組員聯絡上之後：** 請對方先讀本檔和 SETUP.md，接手 B，從 L1–L4 開始。

---

## 4. 資源預算

### nano5（每人約 400 SU，大約等於 480 GPU 分鐘）

| 階段 | 步驟 | GPU 分鐘上限 |
|---|---|---|
| 環境與快取 | N0–N2 | 20 |
| 通訊 | N3 | 8 |
| 主實驗 | N4–N7 | 36 |
| 消融 | N8–N10 | 30 |
| 剖析 | N11–N12 | 18 |
| **合計** | | **112**（計費以實際執行時間為準，通常更少） |
| 重跑緩衝 | | 再加約 100 |

**一個帳號就夠用**（約用掉一半）。另一個帳號留作備援。

**nano5 儲存空間：** venv 約 12 GB、模型約 16 GB、快取約 2 GB，合計約 30 GB，上限 100 GB。

**nano5 規則：** 只能用 `dev` partition；每個作業最多 2 張 GPU、4 個 CPU core、6 分鐘；同時最多 2 個作業；禁止 `salloc`；**不在登入節點運算**；**不在 nano5 上執行任何 AI coding agent**；作業一次送一個，不用迴圈連續提交。`scripts/nano5/submit.sh` 會自動檢查前三項限制，並把每次提交記錄到 `results/logs/su_ledger.csv`。

### server1（共用、沒有額度限制）

實驗本身約 2–3 小時。只使用閒置的 4090（`pick_gpus.sh` 會自動挑選），並在 tmux 裡執行。

---

## 5. 排程（今天 11/24）

| 日期 | 實驗 | 論文 |
|---|---|---|
| **11/24（二）** | L1–L4 本機 smoke test；nano5 N1–N4（安裝、下載模型） | 讀過 `main.tex` 骨架 |
| **11/25（三）** | nano5 N0–N3；確認 server1 恢復後做 S1 | §2 Background 補上通訊數字 |
| **11/26（四）** | nano5 N4–N10；server1 topo、allreduce、main | |
| **11/27（五）** | nano5 N11–N12；server1 ablate、profile；**server1 的最後期限** | 第一次跑分析與畫圖 |
| **11/28（六）** | 補跑缺漏 | §4 Evaluation 依圖撰寫 |
| **11/29（日）** | **凍結數據** | §1 Intro、§5 Related 定稿 |
| **11/30（一）** | | Abstract、Conclusion，**全文初稿** |
| 12/1–12/4 | 在乾淨環境照 SETUP.md 重現一張圖 | 修稿、對照檢查清單 |
| **12/5（五）** | | 送出 |

---

## 6. Repo 結構

```
harness/bench_tp.py         vLLM 離線 benchmark（TP、消融開關、即時寫入 CSV、軟性時限）
harness/allreduce_bench.py  NCCL all-reduce 微基準（torch.distributed），同時記錄 P2P 狀態
configs/                    sweep 設定（main、profile、本機 smoke）、平台峰值規格
env/                        各平台環境變數、requirements.txt（vLLM 版本鎖定）
scripts/nano5/              steps.sh（N0–N12）→ submit.sh（srun 包裝與限制檢查）→ job.sh（計算節點）
scripts/server1/            run_all.sh（分階段執行）、pick_gpus.sh（只挑閒置的 4090）
scripts/local/              smoke.sh（WSL2 + 4060）
analysis/                   model.py（擬合與預測）、nsys_breakdown.py、make_figures.py
results/raw/                原始 CSV 與 meta.json（只新增、不修改）
results/logs/               作業 log、GPU 快照、SU 記錄
figures/                    由 make_figures.py 產生
```

---

## 7. 數據與論文規範

1. **原始數據只新增、不修改。** 某一批數據作廢時，在 `results/logs/NOTES.md` 寫明檔名與原因，分析時再排除。
2. **論文中的每張圖都由 `analysis/make_figures.py` 產生**，不手動修圖。
3. **每天把 `results/` push 上 GitHub。**
4. 不 commit token、密碼或帳號資訊。
5. 論文規範（詳見 `../template/README.md`）：正文含圖表最多 10 頁（目標 6–8 頁）；**不能有 appendix**；雙盲審查，正文與連結中不得出現姓名、學校或 GitHub 帳號，repo 改用 anonymous.4open.science 的匿名連結；參考文獻要列出全部作者；圖中文字至少 8pt。
6. **論文中的紅字**：`[TBD]` 是待填的數字，`[TODO: …]` 是待寫的段落。送出前 PDF 中不能留下任何紅字。
7. 使用生成式 AI 的揭露已寫在 Acknowledgments（camera-ready 版才會顯示）。
8. Workload Distribution 在 `sections/workload.tex`，只有 camera-ready 版（`\submissionfalse`）才會出現。
