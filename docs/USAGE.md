# IMR-Fit 實驗架構使用教學 (Step-by-step)

這份文件假設你剛 clone 完 repo，目標是從零跑出五張論文圖。

---

## 0. 環境準備

### 0a. 安裝 Python 套件 (必做，每台機器只需要一次)

> **重要：** `pip install -r requirements.txt` 必須在 conda env 裡面跑，
> 而且要確認裝進去了。下面逐步驗證。

```bash
conda activate imrfit            # 確認 env 正確

# 安裝所有相依套件
pip install -r requirements.txt

# 專門裝幾個容易被漏掉的套件（明確指定比 requirements.txt 更可靠）
pip install datasets>=2.18 sentence-transformers faiss-cpu numpy
```

如果你遇到網路問題，`datasets` 可以這樣安裝：

```bash
pip install "datasets>=2.18"     # 注意版本號，舊版不支援 wikimedia/wikipedia
```

### 0b. 驗證套件都裝好了

**必做**！一個一個確認，哪個失敗就補裝：

```bash
python -c "
import numpy;              print('numpy        ✓', numpy.__version__)
import sentence_transformers; print('sentence-transformers ✓')
import faiss;              print('faiss        ✓')
import datasets;           print('datasets     ✓', datasets.__version__)
import matplotlib;         print('matplotlib   ✓')
import torch;              print('torch        ✓', torch.__version__)
"
```

每一行都要印出 `✓`。如果哪行報 `ModuleNotFoundError`，就：

```bash
pip install <那個套件名>
```

常見問題：
- `No module named 'datasets'` → `pip install "datasets>=2.18"`
- `No module named 'sentence_transformers'` → `pip install sentence-transformers`
- `No module named 'faiss'` → `pip install faiss-cpu`
- `No module named 'numpy'` → `pip install numpy`（理論上 PyTorch 會帶進來，但有時候 conda env 沒同步）

### 0c. 建立 HDD/SSD 掛載點 (一次性 — 重開機也要)

```bash
sudo mkdir -p /mnt/hdd /mnt/ssd
sudo chown $USER:$USER /mnt/hdd /mnt/ssd
```

如果你還沒有真的 SSD/HDD 分區，可以先用普通目錄模擬（學術實驗用，I/O trace 仍然有效）：

```bash
mkdir -p ~/scratch/hdd ~/scratch/ssd
sudo mount --bind ~/scratch/hdd /mnt/hdd
sudo mount --bind ~/scratch/ssd /mnt/ssd
```

---

## 1. 建立 corpus (Module 1)

> **注意：Step 0 的套件驗證必須先通過才能繼續。**
> corpus 建好之後才能跑 Step 2（workload），順序不能顛倒。

### 1a. 完整版 — 真的下 Wikipedia (慢，~ 60-90 分鐘)

```bash
python -m corpus.build_corpus \
    --hdd-root /mnt/hdd/wiki_corpus \
    --ssd-root /mnt/ssd/faiss_index \
    --target-gb 20
```

如果沒有 `datasets` 套件，會出現：
```
[corpus] HF dataset unavailable — 'datasets' package is not installed
[corpus] Switching to synthetic fallback
```
這表示它自動切到合成資料了，不會 crash。但要跑真實 Wikipedia 就請先裝：`pip install "datasets>=2.18"`

* 文字 chunk 切 512 tokens，128 MB / shard。
* 圖片 100 KB - 5 MB 不等 (雙峰 Z(b))。
* `manifest.jsonl` 紀錄 (chunk_id, kind, shard, offset, size, image_path)。
* Embedding (MiniLM-L6) 同時跑，FAISS IndexIVFFlat 寫到 SSD。

### 1b. 開發 / CI — 用合成資料 (~ 2 分鐘，不需要網路或 datasets 套件)

```bash
python -m corpus.build_corpus \
    --hdd-root /mnt/hdd/wiki_corpus \
    --ssd-root /mnt/ssd/faiss_index \
    --target-gb 4 \
    --synthetic
```

合成資料用 deterministic Zipf vocabulary，跟真實 Wikipedia *結構上* 完全一樣 (shard 大小、雙峰 image 大小)，後續 pipeline 看不出差別。離線、無 GPU、CI 環境都能跑。`--synthetic` 明確指定，就算 `datasets` 沒裝也絕對不會 crash。

### 1c. 驗證

```bash
python -m corpus.verify_corpus
```

輸出範例 (要看到 `bimodal_z_distribution: true`):

```json
{
  "manifest_rows": 12345,
  "shards": 156,
  "shard_avg_mb": 127.8,
  "shard_size_target_mb": 128,
  "text_mean_bytes": 2061,
  "image_mean_bytes": 968512,
  "bimodal_z_distribution": true,
  "index_status": "ok",
  "index_vectors": 12345
}
```

---

## 2. 跑 RAG workload + 產生 cold-tier trace (Modules 2 + 3)

### 2a. 三個 scenario 一次跑完 (預設)

```bash
python run_experiment.py --scenario all
```

每個 scenario 會：
1. 啟動 `TieredStorageSimulator`，SSD cache = 15% 的 corpus。
2. 啟動 `RAGQueryEngine`，掛上 Qwen2-VL-2B Q4 (n_gpu_layers=28, n_ctx=2048)。
3. 執行 300 個 queries (預設可用 `--queries N` 改)。
4. 每個 cache miss 寫一筆到 `traces/scenario_<x>.jsonl`。

### 2b. 沒有 GPU / 想加速時：跳過 LLM

```bash
python run_experiment.py --scenario all --skip-llm
```

只做 retrieval，不做生成。產生的 trace 跟有 LLM 的版本 *I/O pattern 完全相同* (LLM 不會打 cold tier)，所以分析結果一致。

### 2c. 觀察 scenario 個別表現

```bash
python run_experiment.py --scenario c --queries 500    # 只跑 C，多打點
```

| Scenario | 預期 trace 特徵                          | 預期 IMR-Fit 結果 |
|----------|------------------------------------------|-------------------|
| A — Bursty Frequent | 高 F(b), 高 R(b), 寫入 = 0    | Top track 候選聚集 |
| B — Sequential Scan | 低 F(b), 高 Q(b), 高 Z(b)     | Bottom track 候選 |
| C — Mixed + Writes  | 四個維度都有 variance          | 區分度最高，killer figure |

---

## 3. 分析 trace → 4D + S(b) (Module 4)

orchestrator 會自動跑分析，但你也可以單獨叫起：

```bash
python -m imrfit.analyzer \
    --trace traces/scenario_c.jsonl \
    --out-dir results/standalone/scenario_c \
    --grid-search
```

輸出：

* `placement_decisions.jsonl` — 每個 block 一行 `{block_id, F, Q, Z, R, S, placement}`
* `summary.json` — 全域分佈統計、S(b) variance (這是論文要強調的數字)
* `grid_search.json` — top-10 weight combinations 按 RMW reduction surrogate 排序

### 3a. Weight sensitivity

要找最佳 weights 不需要把整個實驗重跑 — analyzer 把 trace 聚合後，可以快速 sweep:

```bash
python -m imrfit.analyzer --trace traces/scenario_c.jsonl --grid-search
```

Grid 是 `{0.1, 0.2, 0.4, 0.6, 0.8}` 四個維度，受 `sum=1` 限制，大約幾十個 candidates。

---

## 4. Replay 對比四個策略 (Module 5)

orchestrator 也會自動跑 replay，但你想單獨重跑某個 trace:

```bash
python -m imrsim.replay \
    --trace traces/scenario_c.jsonl \
    --scenario C \
    --out results/standalone/scenario_c_replay.json \
    --backend python \
    --epoch-io 5000 \
    --budget 32
```

`--backend python` (預設) 用純軟體 RMW penalty model；`--backend kernel` 嘗試對 `/dev/mapper/imrsim` 真機跑 (目前是 stub，見下面 [DRIVER_INTEGRATION.md](DRIVER_INTEGRATION.md))。

四個策略：

| Strategy | 來源 | 模型 |
|----------|------|------|
| `cmr_baseline` | 假裝沒有 RMW penalty | 上界 |
| `naive_imr`    | random 50/50 | 下界 |
| `tracklace`    | F(b) 單一維度 | 1-D baseline |
| `imrfit`       | 4-D S(b) + budget-bounded migration | 你的方法 |

---

## 5. 出圖 (Module 6)

```bash
python plot_results.py --results-dir results/latest
```

5 張 300 DPI PNG 落在 `results/latest/figures/`:

| 檔名 | 內容 |
|------|------|
| `figure1_score_violin.png` | S(b) 三 scenario violin plot — 強調 RAG variance >> ResNet variance |
| `figure2_throughput_vs_epoch.png` | 4 條曲線收斂，IMR-Fit 逼近 CMR |
| `figure3_rmw_bars.png` | scenario × strategy 的 RMW count |
| `figure4_displacement.png` | D(e) 隨 migration epoch 遞減 |
| `figure5_z_bimodal.png` | text ~2KB / image ~1MB 雙峰 — 多模態的獨特貢獻 |

---

## 6. 跨多次實驗的 reproducibility

```bash
# 完整實驗一次，看 results/run_20260427T140000/
python run_experiment.py --scenario all

# 跑相同設定但不同 seed，比較跨實驗 variance
python run_experiment.py --scenario all --queries 300

# results/run_<UTC>/ 自動建好；results/latest -> 最新一次
```

每個 run 目錄都包含完整的 `summary.json`，記錄了所有 CLI 參數，所以可以從一個目錄完整重現一次實驗。

---

## 7. 常用 flag cheatsheet

| 情境 | 指令 |
|------|------|
| 第一次安裝快速跑通 | `python run_experiment.py --scenario a --skip-llm --synthetic --subset 0.05 --queries 50` |
| 寫論文的完整 run | `python run_experiment.py --scenario all --grid-search` |
| Corpus 已建好，重跑分析 | `python run_experiment.py --skip-corpus --scenario all` |
| Trace 已有，只想換 weights 重 replay | `python run_experiment.py --analyze-only --w-freq 0.5 --w-seq 0.2 --w-size 0.1 --w-rec 0.2` |
| 強制走 Python fallback (不用 IMRSim 內核) | `python run_experiment.py --analyze-only --fallback-imrsim` |
| 細看 weight sensitivity | `python run_experiment.py --analyze-only --grid-search` |

---

## 8. 除錯與常見問題

### 8a. `sentence-transformers unavailable` warning

不影響 pipeline 執行 — 自動 fallback 到 deterministic hash embeddings。要消掉：

```bash
pip install sentence-transformers
```

### 8b. `llama-cpp-python` 編譯失敗

CUDA build 需要：

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --no-cache-dir --force-reinstall llama-cpp-python
```

如果還是不行，用 `--skip-llm` 先跑通其他模組。LLM 對 cold-tier I/O pattern 沒影響。

### 8c. FAISS OOM

超過 ~1M chunks 時 IndexIVFFlat 訓練可能吃掉太多 RAM。對策：

* `corpus/build_corpus.py:_build_faiss_index()` 中 `nlist` 已經是 `sqrt(n)`；可以再小一點。
* 大型 corpus 改用 `IndexIVFPQ` (壓縮 vectors)。

### 8d. `/dev/mapper/imrsim` 不存在

這是預期的 — `imrsim/replay.py` 的 kernel backend 會自動 fall back 到 Python。

### 8e. Trace 過大

Scenario 跑越久，trace 越大 (一個 record ~ 150 bytes)。10 萬個 cache miss = 15 MB。沒問題的數量級。如果真的爆掉，可以加 `gzip` 寫到 `.jsonl.gz` (在 `storage/tier_simulator.py:_TraceWriter` 改一下即可)。

---

## 9. 怎麼知道結果是合理的？

跑完 `python run_experiment.py --scenario all` 後檢查：

```bash
python3 -c "
import json
for s in ['scenario_a', 'scenario_b', 'scenario_c']:
    a = json.load(open(f'results/latest/03_{s}_summary.json'))
    print(f'{s}: S variance = {a[\"score_variance\"][\"variance\"]:.4f}')
"
```

期待：
* Scenario A variance < Scenario C variance (mixed workload 區分度最高)
* `figure3_rmw_bars.png` 應該看到 `imrfit < tracklace < naive_imr`，且 `cmr_baseline = 0`
* `figure4_displacement.png` 應該是單調遞減的曲線

如果 `imrfit` 沒贏 `tracklace`，多半是：
* 寫入 chunks 太少 — 加大 `--queries` 或調高 scenario C 的 `write_ratio`
* `--budget` 太小，每個 epoch 移不夠快 — 改大
* Weights 不對 — 跑 `--grid-search` 找新組合

---

未來真的接上 IMR HDD driver 時，請看 [DRIVER_INTEGRATION.md](DRIVER_INTEGRATION.md)。
