# vLLM 張量並行擴展性研究：工作計畫

> **When Does Tensor Parallelism Pay Off? A Communication-Aware Study of vLLM Inference on PCIe and NVLink GPUs**

- 繳交期限：**12/5（五）**。內部目標：**11/30（一）完成全文初稿**
- 論文：repo 內的 `paper/`（由原本的 `template/` 複製進來；ACM SIGPLAN / PPoPP 格式，主檔 `main.tex`）。三人都在這裡改，改不同的 `sections/*.tex` 以減少衝突
- 逐步操作指令：**[SETUP.md](SETUP.md)**（新組員請先讀 SETUP.md 第 0 節）
- 最後更新：11/26

---

## 0. 新組員先讀（10 分鐘）

1. 讀完本檔第 1–4 節，確認自己負責的角色（第 3 節）。
2. 請 repo owner 把你加為 GitHub collaborator，然後照 SETUP.md 第 0 節把 repo 拉到自己的 WSL。
3. 照第 3 節的「第一件事」開始做，做完在群組回報，並更新第 2 節的進度表。
4. **環境已經踩過很多坑（SETUP.md 第 0.2 節）**，安裝請完全照 SETUP.md 的指令，不要自己 `pip install vllm`。

---

## 1. 研究問題

| | 問題 | 對應課程要求 |
|---|---|---|
| **Q1** | vLLM 的張量並行（TP）在 NVLink（H100）與 PCIe（RTX 4090）上擴展效率差多少？差距有多少來自通訊？ | speedup、strong/weak scaling、profiling、ablation |
| **Q2** | 用單卡數據與 all-reduce 微基準校準的「計算 + 記憶體 + 通訊」模型，能多準確預測多卡結果？ | modeling |

**固定條件（所有平台相同）：**

| 項目 | 設定 |
|---|---|
| 模型 | Qwen3-8B（dense，BF16，36 層，hidden 4096，32 Q heads / 8 KV heads） |
| 工作負載 | 輸入 256、輸出 128 tokens；batch 1、4、8、16、32、64；每點 3 次 |
| 軟體 | vLLM **0.32.0+cu129**、torch 2.13.0+cu129、Python 3.11 |
| 取樣 | greedy（temperature 0），PyTorch 原生取樣器（`VLLM_USE_FLASHINFER_SAMPLER=0`） |
| 通訊路徑 | vLLM CUSTOM all-reduce → NCCL。關閉 FlashInfer all-reduce、symmetric-memory all-reduce、all-reduce+RMSNorm 融合 |

**通訊路徑為什麼要固定（論文 Setup 要寫）：** vLLM 0.32 在 H100 上預設會用 FlashInfer 的 all-reduce，並把它和 RMSNorm 融合成一個 kernel。我們把這些關掉，理由有三：(1) 每次 all-reduce 都是獨立操作，可以單獨量測，與模型的 T_comm 假設一致；(2) 兩個平台走同一條路徑，比較才公平；(3) 這些 kernel 需要現場編譯，nano5 的 nvcc 12.4 編不過。設定寫在 `harness/bench_tp.py` 開頭，並記錄在每個 `.meta.json`。

**刪除的項目：** 4060 的數據（只用來開發）、調校類實驗（chunked prefill、prefix caching、量化）、14B 模型、MoE 模型（只在 future work 提一句）。

---

## 2. 目前進度（請隨時更新）

| 項目 | 狀態 | 備註 |
|---|---|---|
| 論文骨架 `paper/main.tex` | ✅ | 各節草稿完成，紅字 `[TBD]` 待填數字 |
| harness、分析程式 | ✅ | `analysis/` 只用假資料測過 |
| nano5 環境 | ✅ | vLLM 0.32.0 cu129、獨立 Python 3.11、FlashInfer 預編譯包 |
| nano5 N0–N2（環境、TP=1/2 smoke） | ✅ | 見下方初步結果 |
| nano5 N3（all-reduce）、N4–N5（主實驗） | ⏳ | 負責人 A |
| nano5 N6–N12（第二輪、消融、Nsight） | ⬜ | 負責人 A |
| server1 環境與實驗 | ⬜ | 負責人 B；**11/27 是最後期限** |
| 本機 WSL 環境對齊到 0.32.0 | ⬜ | 每個人都要（目前本機裝的是 0.34.0） |
| 參考文獻在 DBLP 核對 | ⬜ | 負責人 C |
| 分析、畫圖、填數字 | ⬜ | 負責人 C |

**初步結果（smoke，各 1 次，正式數字以主實驗為準）：**

| H100 | TP=1 | TP=2 | 加速 |
|---|---|---|---|
| decode TPOT，B=1 | 6.61 ms | 4.45 ms | 1.49× |
| decode TPOT，B=64 | 8.11 ms | 5.31 ms | 1.53× |
| prefill，B=1 | 13.2 ms | 17.0 ms | 0.78×（變慢） |
| prefill，B=64 | 386 ms | 263 ms | 1.47× |

- TP=1、B=1 的 decode 約達 74% 記憶體頻寬峰值；B=64 的 prefill 約 700 TFLOPS。
- 引擎啟動約 57 秒，一個主實驗作業約 1.5 分鐘就能跑完，6 分鐘上限很寬裕。
- TP=2 每步 72 次 all-reduce，平均每次多出約 16 µs，主要是延遲而非頻寬。這是論文的主要故事。

---

## 3. 分工（三人）

| | **A：nano5 實驗** | **B：server1 實驗** | **C：分析與寫作** |
|---|---|---|---|
| 主要工作 | nano5 N3–N12、回收數據、harness 除錯 | server1 環境、topo、allreduce、main、ablate、profile | `analysis/` 跑模型與畫圖、Nsight 分類、填 `[TBD]`、統稿 |
| 第一件事 | 跑 N3→N4→N5，push 結果 | **今天確認 server1 能不能用**（SETUP S0），回報驅動版本與閒置 GPU 數 | 本機環境（L1–L3），用現有 nano5 數據跑一次 `analysis/`（SETUP A） |
| 論文章節 | §4.1 Setup、§3 的 harness 部分 | §4 的 PCIe 結果、Limitations | §1、§2、§3 模型、§4 其餘、§5、§6、Abstract |
| 需要的帳號 | nano5（已有） | server1 | GitHub 即可 |

- **介面：** 實驗者只負責把 `results/raw/*.csv`、`*.meta.json`、log push 上 GitHub；C 只從 `results/` 讀資料，不手動改數字。
- **卡住時：** 先把 log 最後 40 行貼到群組，不要重複送作業（nano5 額度用完不補發）。
- **nano5 帳號不共用。** B、C 如果有自己的 nano5 帳號，可以當備援，但環境必須完全照 SETUP N3 建。
- **server1 不能用時：** B 改去支援 C（Related Work、Background 的通訊數字、圖表），論文降級成只用 H100（見第 4 節）。
- 每個人最後要在 `paper/sections/workload.tex` 寫自己的分工（camera-ready 才會顯示）。

---

## 4. 工作分級

| 等級 | 項目 | 產出（論文中的位置） |
|---|---|---|
| **Must** | nano5：N3–N5（all-reduce、TP=1/2 主實驗） | Fig. 1 all-reduce、Fig. 2 strong scaling |
| **Must** | server1：topo、allreduce、main（TP=1/2/4） | Fig. 1、Fig. 2、Fig. 3 weak scaling |
| **Must** | 模型擬合與驗證（`analysis/model.py`） | Fig. 5、Table 3 |
| **Should** | 消融：N8–N10、server1 ablate | Table 2 |
| **Should** | Nsight 剖析：N11–N12、server1 profile | Fig. 4 |
| **Could** | 第二輪主實驗（N6–N7）作跨作業驗證 | 誤差棒、可信度 |
| **Could** | DP 複本對照的討論 | §5 Discussion |

**server1 若到 11/27 仍無法使用：** 論文改成只用 H100（TP=1 對 2 加上模型），在 Limitations 誠實說明缺少 PCIe 對照。這是降級方案，不是失敗。

---

## 5. 資源預算

### nano5（每人約 400 SU ≈ 480 GPU 分鐘）

| 階段 | 步驟 | GPU 分鐘上限 | 狀態 |
|---|---|---|---|
| 環境與除錯 | N0–N2（含失敗重跑） | ~30 | 已用 |
| 通訊 | N3 | 8 | |
| 主實驗 | N4–N7 | 36 | |
| 消融 | N8–N10 | 30 | |
| 剖析 | N11–N12 | 18 | |
| **合計** | | **~120** | 實際計費以執行時間為準，通常更少 |

`wallet` 指令在我們的環境會印出亂碼，所以用量以 `results/logs/su_ledger.csv`（每次提交的時間上限）估計最壞情況。

**nano5 儲存空間：** 約 40 GB（venv + FlashInfer 預編譯包、模型 16 GB、Python、快取），上限 100 GB。

**nano5 規則：** 只能用 `dev` partition；每個作業最多 2 張 GPU、4 個 CPU core、6 分鐘；同時最多 2 個作業；禁止 `salloc`；**登入節點不跑 GPU 程式**（可以編譯、下載、送作業）；**不在 nano5 上執行任何 AI coding agent**；作業一次送一個。`scripts/nano5/submit.sh` 會自動檢查前三項限制。

### server1（共用、沒有額度限制）

實驗本身約 2–3 小時。只使用閒置的 4090（`pick_gpus.sh` 會自動挑選），並在 tmux 裡執行。

---

## 6. 排程（從 11/26 起）

| 日期 | A（nano5） | B（server1） | C（分析/寫作） |
|---|---|---|---|
| **11/26（四）** | N3–N5 | S0 確認可用性、S1 環境 | L1–L3、讀論文骨架、核對參考文獻 |
| **11/27（五）** | N6–N10 | topo、allreduce、main；**最後期限** | 用 nano5 數據第一次跑分析與畫圖 |
| **11/28（六）** | N11–N12、補跑 | ablate、profile、補跑 | §2、§3 模型定稿 |
| **11/29（日）** | **凍結數據** | **凍結數據** | §4 依圖撰寫 |
| **11/30（一）** | 寫 §4.1 | 寫 PCIe 結果 | Intro、Abstract、Conclusion，**全文初稿** |
| 12/1–12/4 | 在乾淨環境照 SETUP 重現一張圖 | 同左 | 統稿、對照檢查清單、匿名化 |
| **12/5（五）** | | | 送出 |

---

## 7. Repo 結構

```
harness/bench_tp.py         vLLM 離線 benchmark（TP、消融開關、固定通訊路徑、即時寫入 CSV、軟性時限）
harness/allreduce_bench.py  NCCL all-reduce 微基準（torch.distributed），同時記錄 P2P 狀態
configs/                    sweep 設定（main、profile、本機 smoke）、平台峰值規格
env/                        各平台環境變數、requirements.txt（版本鎖定與安裝指令）
scripts/nano5/              steps.sh（N0–N12）→ submit.sh（srun 包裝與限制檢查）→ job.sh（計算節點）
scripts/server1/            run_all.sh（分階段執行）、pick_gpus.sh（只挑閒置的 4090）
scripts/local/              smoke.sh（WSL2 + 4060）
analysis/                   model.py（擬合與預測）、nsys_breakdown.py、make_figures.py
results/raw/                原始 CSV 與 meta.json（只新增、不修改）
results/logs/               作業 log、GPU 快照、SU 記錄、NOTES.md
figures/                    由 make_figures.py 產生
paper/                      論文 LaTeX（main.tex、sections/、references.bib、figures/）
```

**資料標籤：** `tag=smoke` 是測試資料，分析不採用；正式資料是 `main`、`no_car`、`eager`、`profile`。

---

## 8. 數據與論文規範

1. **原始數據只新增、不修改。** 某一批數據作廢時，在 `results/logs/NOTES.md` 寫明檔名與原因，分析時再排除。
2. **論文中的每張圖都由 `analysis/make_figures.py` 產生**，不手動修圖。
3. **實驗跑完當天就把 `results/` push 上 GitHub。** push 前先 `git pull --rebase`，避免互相覆蓋。
4. 不 commit token、密碼或帳號資訊。
5. 論文規範（詳見 `paper/README.md`）：正文含圖表最多 10 頁（目標 6–8 頁）；**不能有 appendix**；雙盲審查，正文與連結中不得出現姓名、學校或 GitHub 帳號，repo 改用 anonymous.4open.science 的匿名連結；參考文獻要列出全部作者；圖中文字至少 8pt。
6. **論文中的紅字**：`[TBD]` 是待填的數字，`[TODO: …]` 是待寫的段落。送出前 PDF 中不能留下任何紅字。
7. 使用生成式 AI 的揭露已寫在 Acknowledgments（camera-ready 版才會顯示）。
8. Workload Distribution 在 `paper/sections/workload.tex`，只有 camera-ready 版（`\submissionfalse`）才會出現。
