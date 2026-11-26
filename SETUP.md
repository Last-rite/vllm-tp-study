# SETUP：逐步指令（本機 / nano5 / server1 / 分析）

每一步都附預期結果；結果不符就停下來，把 log 最後 40 行貼到群組，先處理再往下。
`<...>` 是要換成你自己的值。

| 角色 | 要做的章節 |
|---|---|
| 所有人 | 0、L1–L3 |
| A（nano5） | N（環境已建好，從 N5 的 N3 繼續） |
| B（server1） | S0–S3 |
| C（分析/寫作） | A、P |

---

## 0. 新組員快速上手

### 0.1 取得 repo

repo 是 private：https://github.com/Last-rite/vllm-tp-study 。請 owner 到 Settings → Collaborators 把你加進來，接受邀請後在 WSL 裡：

```bash
git config --global user.name "<你的名字>"
git config --global user.email "<你的 GitHub email>"
git clone https://github.com/Last-rite/vllm-tp-study.git ~/vllm-tp-study
cd ~/vllm-tp-study && git log --oneline | head -5
```

密碼欄位要貼 GitHub **classic** personal access token（勾 `repo` 權限）。不要把 token 存在共用主機（nano5、server1）上。

**每天開工：** `git pull --rebase`。**收工：** `git add <你改的檔案> && git commit -m "..." && git pull --rebase && git push`。只 add 自己負責的檔案，不要 `git add -A`。

### 0.2 已知的坑（全部已處理，照本文件做就不會遇到）

| 問題 | 原因 | 處理方式 |
|---|---|---|
| `pip install vllm` 裝到 0.34.0，nano5 跑不起來 | PyPI 最新版是 CUDA 13 wheel，需要 R580 以上驅動；nano5 是 550（CUDA 12.4） | 固定裝 **vLLM 0.32.0 cu129**（最後一個有 CUDA 12.9 wheel 的版本），見 L3/N3/S1 |
| srun 作業裡 `No module named 'torch'` / `importlib.metadata` | nano5 計算節點的系統 Python 是 3.6，登入節點是 3.11；venv 連到系統 Python 就會出錯 | venv 改用放在 `$STORE` 的 uv 獨立 Python 3.11（N3） |
| `nvcc fatal: Unknown option '--compress-mode=size'` | FlashInfer 會現場編譯 kernel，nano5 的 nvcc 12.4 太舊 | `bench_tp.py` 改用 PyTorch 取樣器、關閉 FlashInfer all-reduce 與 AR+RMSNorm 融合；另外裝 FlashInfer 預編譯包（`vllm download-kernels`） |
| `vllm download-kernels` 中斷後 import flashinfer 出錯 | `flashinfer-jit-cache` 只裝了一半 | `uv pip uninstall flashinfer-jit-cache flashinfer-cubin`，刪掉殘留資料夾，在 tmux 裡重跑 |
| `wallet` 印出亂碼 | 系統指令與環境不相容（未解） | 用 `results/logs/su_ledger.csv` 估計用量；需要時問助教 |
| 登入節點的規則 | 可以下載、編譯、送作業，**不能跑 GPU 程式** | 檢查版本只用 `importlib.metadata`，不要 `import vllm` |

---

## L. 本機（Windows 筆電 + RTX 4060）

vLLM 不支援原生 Windows，所以本機一律在 **WSL2 Ubuntu** 裡執行。本機只用來測通程式，數據不放進論文。

### L1. 安裝 WSL2（只需做一次）

用系統管理員身分開啟 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
# 依指示重新開機，然後設定 Ubuntu 的使用者名稱與密碼
```

先把 Windows 的 NVIDIA 驅動更新到最新版（用 GeForce Experience 或 NVIDIA App）。**不要**在 WSL 裡安裝 Linux 版 NVIDIA 驅動。

開啟 Ubuntu 終端機：

```bash
nvidia-smi                      # 預期：看到 GeForce RTX 4060 Laptop GPU
sudo apt update && sudo apt install -y python3-venv python3-pip git tmux rsync
python3 --version               # 預期：3.12.x
```

### L2. 把 repo 放進 WSL

新組員：照 0.1 用 `git clone`。在 WSL 的檔案系統（`~/`）裡執行，比在 `/mnt/c` 快很多。

（已完成，僅供參考）repo 最初是從 Windows 資料夾複製進 WSL，再 `git init` 後 push 到 GitHub。**現在以 GitHub 上的版本為準**，Windows 資料夾裡的舊檔不要再複製回 repo。

同儕互評版的論文要用匿名連結（anonymous.4open.science），不要直接放 GitHub 連結。

### L3. Python 環境

版本固定為 **vLLM 0.32.0（CUDA 12.9 版）**。不要直接 `pip install vllm`：PyPI 上最新版是 CUDA 13 的 wheel，需要 R580 以上驅動，nano5 的驅動是 550（CUDA 12.4），裝了也跑不起來。

```bash
cd ~/vllm-tp-study
source env/local.env.sh
python3 -m venv "$VENV" && source "$VENV/bin/activate"
pip install -U pip uv
uv pip install --no-cache "vllm==0.32.0" --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 --torch-backend=cu129
uv pip install pandas matplotlib
python -c "from importlib.metadata import version as v; print('torch', v('torch'), 'vllm', v('vllm'))"
# 預期：torch 2.13.0+cu129 vllm 0.32.0+cu129
# 若之前裝過 0.34.0：先 rm -rf "$VENV"，再從 python3 -m venv 那行重做
du -sh "$STORE"                        # 預期：約 10–12 GB
```

### L4. 本機 smoke test

```bash
source env/local.env.sh
hf download Qwen/Qwen3-0.6B            # 約 1.5 GB；舊版指令是 huggingface-cli download
bash scripts/local/smoke.sh
```

預期結果：
- 先出現 `engine ready in ...s`，接著出現多行 `B=   1 rep=0 prefill=... tpot=...`，最後是 `done -> results/raw/rtx4060_Qwen3-0.6B_tp1_smoke_....csv`
- `results/raw/` 底下有 `.csv` 與 `.meta.json`

常見問題：
- **CUDA out of memory：** 把 `configs/smoke_local.json` 的 `gpu_memory_utilization` 改成 0.7，並關掉其他使用 GPU 的程式。
- **WSL 記憶體不足而被 kill：** 在 Windows 的 `C:\Users\USER\.wslconfig` 加入 `[wsl2]` 與 `memory=12GB`，再執行 `wsl --shutdown`。

### L5. 用假資料測試畫圖流程（選做，1 分鐘）

```bash
python analysis/model.py      # smoke 數據只有 TP=1，會印出 MAPE；有 WARNING 屬正常
```

---

## N. nano5（H100）

### 規則（每次都要遵守）

- 登入節點只做：安裝套件、下載模型、傳檔、`srun` 提交作業。**不在登入節點跑任何運算或測試**。
- **不在 nano5 上執行任何 AI coding agent**（Claude Code、Codex、Gemini CLI 等）。程式一律在本機改好再傳上去。
- 作業一律用 `scripts/nano5/steps.sh` 一次送一個，**不寫迴圈連續提交**。同時最多 2 個。
- 每個作業結束後看 log 與 `wallet`，確認無誤再送下一個。

### N1. 登入與檢查空間

```bash
ssh <帳號>@<nano5 登入節點位址>        # 位址依課程公告
du -sh ~                               # 目前用量（上限 100 GB）
module avail 2>&1 | grep -i -E "cuda|python|conda" | head
python3 --version                      # 需要 3.10 以上
```

若系統的 `python3` 低於 3.10，用 `module load <python 或 miniconda 模組名稱>` 載入較新版本，模組名稱以 `module avail` 顯示的為準。

### N2. 取得程式碼

```bash
git clone https://github.com/<你的帳號>/vllm-tp-study.git ~/vllm-tp-study
# private repo 需要 GitHub personal access token 當密碼。
# 若 git clone 失敗，改在本機 WSL 執行：
#   rsync -av --exclude results ~/vllm-tp-study/ <帳號>@<nano5 位址>:~/vllm-tp-study/
cd ~/vllm-tp-study
```

### N3. 安裝環境（在登入節點，約 10–60 分鐘）

**重點（踩過的坑）：**

1. nano5 驅動是 550.127.08（CUDA 12.4），所以要裝 CUDA 12.9 版的 wheel（vLLM 0.32.0 是最後一個提供 cu129 的版本）。已驗證 H100 可以正常初始化。
2. **計算節點的系統 Python 只有 3.6**，登入節點才有 3.11。用系統 `python3 -m venv` 建的 venv，在登入節點能跑、在 srun 作業裡會用到 3.6 而失敗（`No module named 'importlib.metadata'`）。所以 venv 必須用放在 `$STORE` 的獨立 Python（uv 管理），兩種節點才會用同一份。
3. 登入節點不要執行 GPU 程式；檢查版本只用 `importlib.metadata`，不要 `import vllm`。

```bash
tmux new -s setup                      # 避免 SSH 斷線中斷安裝
cd ~/vllm-tp-study
source env/nano5.env.sh                # 會建立 ~/vtp-store 下的各個資料夾
pip install -U uv                      # 若系統沒有 uv，先用使用者安裝：pip install --user uv

# (a) 獨立 Python 3.11，放在 $STORE（不要用系統 python3）
export UV_PYTHON_INSTALL_DIR="$STORE/python"
uv python install 3.11
PYBIN=$(ls "$STORE"/python/*/bin/python3.11 | head -1)
"$PYBIN" -V

# (b) 用這個 Python 建 venv，安裝固定版本
"$PYBIN" -m venv "$VENV" && source "$VENV/bin/activate"
uv pip install --no-cache "vllm==0.32.0" --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 --torch-backend=cu129
uv pip install pandas matplotlib
du -sh "$STORE"                        # 預期：約 11–12 GB
```

**如果 venv 已經用系統 Python 建好了**（我們的情況），不用重裝，直接改接連結即可：

```bash
export UV_PYTHON_INSTALL_DIR="$STORE/python"
uv python install 3.11
PYBIN=$(ls "$STORE"/python/*/bin/python3.11 | head -1)
ln -sfn "$PYBIN" "$VENV/bin/python3.11"
ln -sfn python3.11 "$VENV/bin/python3"
ln -sfn python3 "$VENV/bin/python"
sed -i "s|^home = .*|home = $(dirname "$PYBIN")|" "$VENV/pyvenv.cfg"
```

**驗證（兩個節點都要通過）：**

```bash
# 登入節點（只讀套件資訊，不碰 GPU）
"$VENV/bin/python" -c "import sys; print(sys.version.split()[0]); from importlib.metadata import version as v; print('torch', v('torch'), 'vllm', v('vllm'))"
# 計算節點（約 1 GPU 分鐘）
srun -A $SLURM_ACCT -p dev -N1 -n1 -c4 --gpus-per-node=1 -t 00:01:00 bash -c '
source env/nano5.env.sh
readlink -f "$VENV/bin/python"
"$VENV/bin/python" -c "import sys; print(sys.version.split()[0]); import torch; print(torch.__version__, torch.cuda.is_available())"'
```

預期：兩邊都是 Python 3.11.x、`torch 2.13.0+cu129 vllm 0.32.0+cu129`，計算節點最後印出 `True`。

### N4. 下載模型（在登入節點，約 16 GB）

```bash
source env/nano5.env.sh
hf download Qwen/Qwen3-8B --local-dir "$HF_HOME/models/Qwen3-8B"
ls "$HF_HOME/models/Qwen3-8B"          # 預期：config.json、*.safetensors、tokenizer 檔案
du -sh "$STORE"                        # 預期：約 28–30 GB
```

### N5. 依序送出作業

**目前進度：N0–N2 已通過（11/25）。** 接著從 N3 開始。每個作業實際只跑約 1–2 分鐘，可以用 `&&` 串接，前一個失敗就自動停下：

```bash
bash scripts/nano5/steps.sh N3 && bash scripts/nano5/steps.sh N4 && bash scripts/nano5/steps.sh N5
```

**每次跑完都要檢查通訊路徑（TP=2 的作業）：**

```bash
LOG=$(ls -t results/logs/nano5_*_bench.log | head -1)
grep -n "all-reduce backends\|custom fusions\|engine ready\|B=" "$LOG"
# 預期：Using ['CUSTOM', 'PYNCCL']（N8 no_car 會只剩 PYNCCL），且沒有 "Enabled custom fusions: allreduce_rms"
```

每個指令都在登入節點的 `tmux` 裡執行（`tmux attach -t setup`）。`srun` 會等待作業結束，結束時會印出剩餘額度。

```bash
cd ~/vllm-tp-study
bash scripts/nano5/steps.sh list       # 列出所有步驟
bash scripts/nano5/steps.sh N0         # 環境檢查
```

| 步驟 | 內容 | GPU 數 | 時限 | GPU 分鐘上限 | 看什麼 |
|---|---|---|---|---|---|
| N0 | 環境檢查（驅動、GPU、torch/vLLM、nsys） | 1 | 2 分 | 2 | 驅動版本、`nvidia-smi topo -m` 是否顯示 NV#、nsys 是否找得到 |
| N1 | TP=1 smoke（batch 1、64 各 1 次），建立編譯快取 | 1 | 6 分 | 6 | `engine ready in` 幾秒；是否寫出 CSV ✅ 57 s |
| N2 | TP=2 smoke | 2 | 6 分 | 12 | log 中 `Using ['CUSTOM', 'PYNCCL'] all-reduce backends` ✅ |
| N3 | all-reduce 測試（1 KB–256 MB） | 2 | 4 分 | 8 | `peer access` 應為 True；大訊息頻寬 |
| N4 | 主實驗 TP=1 | 1 | 6 分 | 6 | 6 種 batch × 3 次是否都完成 |
| N5 | 主實驗 TP=2 | 2 | 6 分 | 12 | 同上 |
| N6 | 主實驗 TP=1（第二次，跨作業驗證） | 1 | 6 分 | 6 | 與 N4 差距應在數 % 內 |
| N7 | 主實驗 TP=2（第二次） | 2 | 6 分 | 12 | 與 N5 比較 |
| N8 | 消融：關閉 custom all-reduce | 2 | 6 分 | 12 | |
| N9 | 消融：關閉 CUDA graphs（eager） | 2 | 6 分 | 12 | |
| N10 | 消融參考：TP=1 eager | 1 | 6 分 | 6 | |
| N11 | Nsight 剖析 TP=2 | 2 | 6 分 | 12 | 產生 `results/nsys/*_cuda_gpu_kern_sum.csv` |
| N12 | Nsight 剖析 TP=1 | 1 | 6 分 | 6 | 同上 |
| | **合計上限** | | | **112** | 一個帳號約有 480 |

**每個作業結束後檢查：**

```bash
ls -t results/logs/ | head -3
tail -n 25 results/logs/$(ls -t results/logs/ | grep nano5 | head -1)
wc -l results/raw/h100_*.csv           # 每個主實驗 CSV 應有 1 + 18 行
```

**依結果決定下一步：**
- **N1 在寫出任何資料前就被 kill：** 冷啟動超過 5.5 分鐘。再跑一次 N1（快取已建立，第二次會快很多）。若仍失敗，看 log 中時間花在載入權重還是編譯，再告訴我。
- **log 出現 `SKIP batch`：** KV cache 放不下該 batch，屬正常現象，已記錄在 log。
- **N0 找不到 nsys：** N11、N12 跳過，論文改用消融結果說明通訊成本。
- **任何錯誤：** 先不要重送。把 log 最後 40 行貼給我，或在本機處理。

### N6. 把結果拿回 GitHub

方法一（推薦）：直接在 nano5 push（每次輸入 token，不要存起來）：

```bash
cd ~/vllm-tp-study
git add results/raw results/logs && git commit -m "nano5: N3-N5 results" && git pull --rebase && git push
```

方法二：在本機 WSL 用 rsync 拉回來再 push：

```bash
cd ~/vllm-tp-study
rsync -av <帳號>@<nano5 位址>:~/vllm-tp-study/results/ ./results/
git add results && git commit -m "nano5 results" && git push
```

`.nsys-rep` 檔案很大，已由 `.gitignore` 排除。拿回本機後，就可以在 nano5 上執行 `rm results/nsys/*.nsys-rep` 釋放空間。

---

## S. server1（4 × RTX 4090）

### S0. 確認能不能用（5 分鐘，11/26 就要做）

```bash
ssh <帳號>@<server1 位址>
nvidia-smi                             # 記下：驅動版本、CUDA 版本、4090 有幾張閒置（Memory-Usage 低、無 process）
nvidia-smi topo -m                     # 4090 之間應該是 PHB/SYS/NODE（PCIe），不會有 NV#
python3 --version; df -h /tmp2         # Python 版本；/tmp2 剩餘空間（需要約 40 GB）
which nvcc nsys
```

把輸出貼到群組。判斷標準：

- **驅動 ≥ 525**：可以跑 cu129 wheel（CUDA 12 的 minor version compatibility）。低於這個版本就停下來討論，**不要升級驅動**（共用機器）。
- **至少 2 張 4090 閒置**：可以開始。TP=4 需要 4 張同時閒置，之後挑深夜跑。
- server1 若到 11/27 仍無法使用，回報後改去支援 C。

### S1. 環境（約 15–60 分鐘，主要是下載）

```bash
tmux new -s vtp
git clone https://github.com/Last-rite/vllm-tp-study.git ~/vllm-tp-study && cd ~/vllm-tp-study
source env/server1.env.sh              # 大檔放 /tmp2/$USER/vtp-store
pip install --user -U uv               # 若沒有 uv
export UV_PYTHON_INSTALL_DIR="$STORE/python"
uv python install 3.11                 # 和 nano5 同樣用獨立 Python 3.11，避免系統 Python 版本差異
PYBIN=$(ls "$STORE"/python/*/bin/python3.11 | head -1)
"$PYBIN" -m venv "$VENV" && source "$VENV/bin/activate"
uv pip install --no-cache "vllm==0.32.0" --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 --torch-backend=cu129
uv pip install pandas matplotlib
vllm download-kernels                  # FlashInfer 預編譯包，與 nano5 一致；中斷的話見 0.2 的處理方式
python -c "from importlib.metadata import version as v; print('torch', v('torch'), 'vllm', v('vllm'))"
# 預期：torch 2.13.0+cu129 vllm 0.32.0+cu129
hf download Qwen/Qwen3-8B --local-dir "$HF_HOME/models/Qwen3-8B"
du -sh "$STORE"
```

先跑一次 TP=1 smoke 確認可用（約 2 分鐘，會用 1 張閒置 GPU）：

```bash
source scripts/server1/pick_gpus.sh 1
python harness/bench_tp.py --config configs/sweep_main.json --model "$HF_HOME/models/Qwen3-8B" \
  --tp 1 --platform rtx4090 --tag smoke --batches 1,64 --reps 1
```

預期最後印出 `done -> results/raw/rtx4090_Qwen3-8B_tp1_smoke_*.csv`。4090 有 24 GB，batch 64 應該放得下；若出現 `SKIP batch` 屬正常。

### S2. 執行（一次一個階段，每階段跑完都看一下結果）

```bash
nvidia-smi                                    # 先確認有沒有其他人正在使用
bash scripts/server1/run_all.sh topo          # 存下 results/logs/server1_topo.txt
bash scripts/server1/run_all.sh allreduce     # 2 卡與 4 卡；印出 peer access
bash scripts/server1/run_all.sh main          # TP=1/2/4 各跑 2 輪，約 30–60 分鐘
bash scripts/server1/run_all.sh ablate
bash scripts/server1/run_all.sh profile
```

- `pick_gpus.sh` 只會挑閒置的 4090。湊不齊需要的卡數時會跳過並印出 `SKIP`，之後再補跑即可。
- TP=4 需要 4 張 4090 同時閒置，建議在深夜或清晨跑。
- **檢查通訊路徑：** TP=2/4 的 log 搜尋 `all-reduce backends`。PCIe 上 vLLM 的 CUSTOM all-reduce 可能會因為沒有 P2P 而自動停用，只剩 `PYNCCL`。這本身就是論文要報告的結果，把那一行貼到群組，並在 `results/logs/NOTES.md` 記下。
- **all-reduce 或 TP=2 卡住不動（hang）：** 先按 `Ctrl-C`，再執行
  `export NCCL_P2P_DISABLE=1` 後重跑，並在論文中註明使用了這個設定。

### S3. 拿回結果（在本機 WSL 執行）

```bash
rsync -av <帳號>@<server1 位址>:~/vllm-tp-study/results/ ./results/
```

跑完後清掉 `/tmp2/$USER/vtp-store`。

---

## A. 分析與畫圖（本機 WSL，C 負責）

```bash
cd ~/vllm-tp-study && git pull --rebase && source env/local.env.sh
python analysis/model.py                                        # 擬合模型 → results/model_params.json、model_predictions.csv
python analysis/nsys_breakdown.py results/nsys/*_cuda_gpu_kern_sum.csv
python analysis/make_figures.py --paper paper
```

- 只讀 `tag` 為 `main`、`no_car`、`eager`、`profile` 的資料；`smoke` 會被忽略。
- 圖會複製到 `paper/figures/`。`main.tex` 裡的佔位框會自動換成真圖，`tab_ablation.tex` 也一樣。
- 接著把 `results/model_params.json` 的數字填進論文 Table 3 與各處的 `[TBD]`。
- `nsys_breakdown.py` 會印出各類別耗時最多的 kernel。若某個耗時很多的 kernel 被歸到 "Other"，就修改檔案開頭的 `PATTERNS`。
- 沒有 server1 數據時，程式應該只畫 H100 的部分；若出錯，貼 traceback 到群組。

---

## P. 論文（`paper/`）

```bash
cd ~/vllm-tp-study/paper
latexmk -pdf main.tex                  # 或上傳整個 paper/ 到 Overleaf 編譯
```

- 分工依 README 第 3 節，每人只改自己的 `sections/*.tex`，減少合併衝突。
- 新的參考文獻加到 `references.bib`，作者要列全，並到 DBLP 核對。
- 送出前搜尋 `\tbd`、`\todo`、`[TBD]`，PDF 裡不能有紅字；確認 `\submissiontrue`（匿名）。
