# 從 mock HDD 換到真實 IMR Driver 的遷移指南

當你拿到 `imr_hdd` driver (一塊真實的 IMR 磁碟，或是修好的 IMRSim kernel module)，這份文件告訴你 **哪些檔案要改、改什麼、怎麼驗證**。

---

## TL;DR

整個架構從一開始就把「儲存層」抽象在 **Tier 介面** 後面，所以：

* **Trace schema 不會變** — 真機跑出來的 trace 跟模擬器一模一樣，下游 (Module 4 analyzer + Module 5 replayer) 完全不用動。
* **改動集中在三個檔案** + 一個 setup script。
* **Fallback 機制保留** — 你可以單機 boot 失敗時隨時 `--fallback-imrsim` 切回去。

---

## 抽象界線

```
┌─────────────────────────────────────────────────────────┐
│ rag/, corpus/                                           │
│  只懂 chunk_id；不知道 disk 長什麼樣                     │
└──────────────────────┬──────────────────────────────────┘
                       │   read(chunk_id) / write(chunk_id, data)
┌──────────────────────▼──────────────────────────────────┐
│ storage.tier_simulator.TieredStorageSimulator           │
│  ↑ 上層 API 不會變                                        │
│  ↓ 下層: 目前是 open(file) 對著 /mnt/hdd                  │
│         → 未來: 要嘛還是 open(file) (IMR HDD 上的 ext4)，  │
│              要嘛是 pwrite(fd, lba) 對著 raw block dev   │
└──────────────────────┬──────────────────────────────────┘
                       │   trace.jsonl  (schema 不變)
┌──────────────────────▼──────────────────────────────────┐
│ imrfit.analyzer / imrsim.replay                         │
│  完全不知道下面是模擬還是真機 ✓                            │
└─────────────────────────────────────────────────────────┘
```

換 driver = 換 `TieredStorageSimulator` 的「下層」+ 把 `imrsim.replay` 的 kernel backend stub 填滿。

---

## 改動清單 (file:line precision)

### 路徑 1 — 真實 IMR HDD 上掛 ext4 / xfs (推薦先做)

最簡單的情境：driver 已經做好，IMR HDD 上面有一個正常的 filesystem mount 在 `/mnt/imr_hdd`。

#### Change 1: 更新預設掛載點

`storage/tier_simulator.py:30`

```python
@dataclass
class TierConfig:
    hdd_root: str = "/mnt/hdd/wiki_corpus"           # 改成 ↓
    hdd_root: str = "/mnt/imr_hdd/wiki_corpus"
```

`corpus/build_corpus.py:55`

```python
DEFAULT_HDD_ROOT = "/mnt/hdd/wiki_corpus"           # 改成 ↓
DEFAULT_HDD_ROOT = "/mnt/imr_hdd/wiki_corpus"
```

或者更乾淨：用環境變數 (見下方 *進階：環境變數化*)。

#### Change 2: LBA 改用 fiemap，不要再用 chunk_index 假算

目前 `storage/tier_simulator.py:_lba_for()` 用 `chunk_index * block_size` 假裝 LBA。在真機上你會想要真正的 LBA，這樣 trace 跟 device 上發生的事情一致。

把 `_lba_for()` 改成：

```python
def _lba_for(self, chunk_id: str) -> int:
    loc = self._manifest._by_id.get(chunk_id)
    if loc is None:
        return self._fallback_lba(chunk_id)

    # 真機: 從 image_path / shard_path 抓 fiemap.
    # ext4 / xfs 都支援 FS_IOC_FIEMAP ioctl.
    path = self._path_for(loc)
    return _fiemap_first_extent_lba(path) + (loc.offset or 0)
```

新增 helper (放在同一檔案):

```python
import struct
import fcntl

_FS_IOC_FIEMAP = 0xC020660B
_FIEMAP_MAX_EXTENTS = 1

def _fiemap_first_extent_lba(path: str) -> int:
    """回傳 file 第一個 extent 在 device 上的 byte offset (LBA)."""
    with open(path, "rb") as fh:
        # struct fiemap: 8 (start) + 8 (length) + 4 (flags)
        #              + 4 (mapped_extents) + 4 (extent_count) + 4 (reserved)
        buf = bytearray(32 + 56 * _FIEMAP_MAX_EXTENTS)
        struct.pack_into(
            "QQIII", buf, 0,
            0,                # fm_start
            1 << 60,          # fm_length (whole file)
            0,                # fm_flags
            0,                # fm_mapped_extents (out)
            _FIEMAP_MAX_EXTENTS,
        )
        fcntl.ioctl(fh.fileno(), _FS_IOC_FIEMAP, buf, True)
        # First extent starts at offset 32 of the buffer.
        # struct fiemap_extent: 8 (logical) + 8 (physical) + 8 (length) ...
        physical = struct.unpack_from("Q", buf, 32 + 8)[0]
        return int(physical)
```

注意：`fiemap` 需要 root 或 `CAP_SYS_RAWIO` (有時候)。如果跑進權限問題，留著舊的 `chunk_index * block_size` 當 fallback：

```python
def _lba_for(self, chunk_id: str) -> int:
    try:
        return self._real_lba(chunk_id)        # fiemap path
    except (OSError, PermissionError):
        return self._synthetic_lba(chunk_id)   # 舊的算法
```

#### Change 3: `fsync_on_write = True`

Mock 模式下我們關掉 fsync (簡單為了快)。真機上要打開：

`storage/tier_simulator.py:33`

```python
fsync_on_write: bool = False                         # 改成 ↓
fsync_on_write: bool = True
```

否則 RMW 行為會被 page cache 吸收，trace 不真實。

---

### 路徑 2 — 直接對 raw IMR block device 操作 (進階)

如果 driver 還沒搭配 filesystem (例如你想直接看 dmsetup 結果)，需要做更深的改動。

#### Change A: 用 raw device 取代 file API

新增 `storage/raw_block_device.py`:

```python
"""Raw block-device adapter for the cold tier.

Replaces the per-file open()/read()/write() calls in TieredStorageSimulator
with pwrite/pread against /dev/imr_hdd at deterministic LBAs.

Layout convention (must match corpus.build_corpus):
    chunk_index N   -> LBA  N * block_size
"""
import os
from typing import Optional


class RawBlockDevice:
    def __init__(self, device_path: str = "/dev/imr_hdd",
                 block_size: int = 128 * 1024 * 1024) -> None:
        self.device_path = device_path
        self.block_size = block_size
        self._fd = os.open(device_path, os.O_RDWR | os.O_DIRECT)

    def read_chunk(self, chunk_index: int, size: int) -> bytes:
        offset = chunk_index * self.block_size
        return os.pread(self._fd, size, offset)

    def write_chunk(self, chunk_index: int, data: bytes) -> None:
        offset = chunk_index * self.block_size
        os.pwrite(self._fd, data, offset)

    def close(self) -> None:
        os.close(self._fd)
```

然後在 `storage/tier_simulator.py` 加 `device: Optional[RawBlockDevice]` 參數，
在 `_read_from_hdd` / `_write_to_hdd` 裡優先走 device。

#### Change B: corpus build 也要改

`corpus/build_corpus.py` 目前寫 jsonl shard。改 raw mode 時要：

1. 直接 `pwrite(fd, jsonl_bytes, chunk_index * block_size)`
2. 在 manifest 中保留 `shard=None`、`offset=0`、用 `chunk_index` 當 LBA key

這個改動比較大，建議先走「路徑 1」(filesystem on top of IMR)。

---

### 路徑 3 — 真正接上 IMRSim kernel module

`imrsim/replay.py:_replay_kernel()` 目前是 stub (有清楚的註解標明)。要填的內容：

```python
def _replay_kernel(self, strategy, trace_path, placement_map, optimal,
                   *, scenario, n_blocks):
    device = "/dev/mapper/imrsim"
    util = os.path.expanduser("~/IMRSim/imrsim_util/imrsim_util")

    # 1. dmsetup load 一個全新的 imrsim target，套用 placement_map
    self._apply_placement_via_dmsetup(device, placement_map)

    # 2. 重置 zone counters
    subprocess.run([util, device, "s", "4"], check=True)

    # 3. 開 fd，逐筆 replay trace
    fd = os.open(device, os.O_RDWR | os.O_DIRECT)
    counters_per_epoch: list[dict] = []
    try:
        with open(trace_path) as fh:
            for i, line in enumerate(fh):
                rec = json.loads(line)
                if rec["op"] == "R":
                    os.pread(fd, rec["size"], rec["lba"])
                else:
                    payload = b"\\x00" * rec["size"]
                    os.pwrite(fd, payload, rec["lba"])

                # epoch 結束 → 抓 zone stats
                if (i + 1) % self.cfg.epoch_io_count == 0:
                    counters_per_epoch.append(self._poll_imrsim(util, device))
                    if strategy == Strategy.IMRFIT:
                        self._apply_migration_round(device, optimal)
    finally:
        os.close(fd)

    return self._counters_to_replay_result(strategy, scenario, n_blocks,
                                           counters_per_epoch)
```

**搭配的兩個 helper** (放同一檔案):

* `_apply_placement_via_dmsetup` — 用 `dmsetup message <dev> 0 set_top <block_id>` 之類的命令告訴 IMRSim 哪些 block 走 top track。確切語法視 driver fork 而定。
* `_poll_imrsim` — wraps `imrsim_util <dev> s 1`，回傳 `DeviceStats`。**這部分已經做好** — 直接呼叫 `imrfit.monitor.IMRSimMonitor.poll()`。

---

## Setup script — `scripts/02_setup_device.sh`

這個檔案在 repo 裡已經存在。把它改成 driver-aware:

```bash
#!/usr/bin/env bash
set -euo pipefail

DEVICE="${IMR_DEVICE:-/dev/sdX}"        # 改成你的真實 device
MOUNT="${IMR_MOUNT:-/mnt/imr_hdd}"

# 1. 確認 driver loaded
lsmod | grep -q imr_hdd || sudo modprobe imr_hdd

# 2. 檢查 device 出現
[ -b "$DEVICE" ] || { echo "device $DEVICE not present"; exit 1; }

# 3. Format (一次性！)
if [ "${1:-}" = "--format" ]; then
    sudo mkfs.ext4 -F "$DEVICE"
fi

# 4. Mount
sudo mkdir -p "$MOUNT"
sudo mount "$DEVICE" "$MOUNT" -o rw,relatime
sudo chown "$USER:$USER" "$MOUNT"

echo "IMR HDD ready at $MOUNT"
```

---

## 進階：環境變數化所有 mount points

為了避免每次換 driver 都改 default 字串，建議讓所有路徑可以從 env 覆蓋。

新增 `storage/paths.py`:

```python
"""Single source of truth for tier mount points."""
import os

HDD_ROOT = os.environ.get("IMRFIT_HDD_ROOT", "/mnt/hdd/wiki_corpus")
SSD_ROOT = os.environ.get("IMRFIT_SSD_ROOT", "/mnt/ssd/faiss_index")
IMR_DEV  = os.environ.get("IMRFIT_IMR_DEVICE", "/dev/mapper/imrsim")
```

然後 `storage/tier_simulator.py`、`corpus/build_corpus.py`、`run_experiment.py` 改成 import 這個 module 的常數。改 driver 時只要：

```bash
export IMRFIT_HDD_ROOT=/mnt/imr_hdd/wiki_corpus
export IMRFIT_IMR_DEVICE=/dev/imr_hdd
python run_experiment.py --scenario all
```

---

## 漸進式驗證計畫 (建議照這個順序做)

每一步都要能跑通才往下走。每步都用同一個小 corpus (4 GB synthetic) 加速 iteration。

| Step | 目標 | 通過標準 |
|------|------|----------|
| 0 | 確認 mock 還能跑 | `python run_experiment.py --skip-llm --synthetic --subset 0.05` 五張圖出來 |
| 1 | driver loaded、`/mnt/imr_hdd` mount 成功 | `mount \| grep imr_hdd`，`touch /mnt/imr_hdd/test` 不報錯 |
| 2 | 改 `hdd_root`，重建 corpus 在 IMR 上 | `verify_corpus` 通過、`bimodal_z_distribution: true` |
| 3 | 跑 scenario A 不開 fiemap | trace 大小、record 數正常 (對比 mock) |
| 4 | 開 fiemap LBA | trace 中的 lba 應該是 raw byte offset on device，*不會*再都是 N * 128 MB 的整數倍 |
| 5 | `fsync_on_write=True`，重跑 scenario C | 寫入 latency 變大；trace 中 W records 應該變多 (沒被 page cache 吃掉) |
| 6 | 拔掉 `--fallback-imrsim`，跑 kernel replay | `imrsim_util` 應該回報 RMW counts > 0 |
| 7 | 對齊驗證 | mock vs 真機在 scenario A 上的 RMW count 應該 trend 一致 (絕對值會差，相對排序不會) |

如果 step 4 的 LBA 變成同一個 device offset (e.g. 全部都是 0)，代表 fiemap call 失敗 — 多半是 EOPNOTSUPP (filesystem 不支援) 或權限不夠。改 setup script `chmod` 或加 `CAP_SYS_RAWIO`。

---

## 不需要改的部分

下列檔案 **完全不用碰** (這就是模組化的好處):

* `imrfit/analyzer.py` — 只吃 trace.jsonl，不在乎來源。
* `imrfit/scorer.py`, `scheduler.py`, `monitor.py` — 早就跟 device 解耦了。
* `imrsim/fallback_simulator.py` — 永遠保留作 baseline 對比與 CI fallback。
* `rag/`, `corpus/` 的核心 streaming logic — 只有預設路徑要改。
* `plot_results.py` — 看的是 results JSON，不在乎 backend。

---

## 驗證 trace schema 真的不變

寫個 unit test 鎖住 schema:

```python
# tests/test_trace_schema.py
import json
from pathlib import Path

REQUIRED = {"ts_ns", "chunk_id", "lba", "size", "op", "scenario", "cache_hit"}

def test_trace_schema_unchanged(tmp_path: Path):
    from storage.tier_simulator import TieredStorageSimulator, TierConfig
    cfg = TierConfig(
        hdd_root=str(tmp_path),
        trace_path=str(tmp_path / "t.jsonl"),
        scenario="X",
    )
    with TieredStorageSimulator(cfg) as sim:
        sim.write("foo", b"x" * 16, kind="text")

    line = (tmp_path / "t.jsonl").read_text().splitlines()[0]
    rec = json.loads(line)
    assert REQUIRED.issubset(rec.keys()), f"missing: {REQUIRED - rec.keys()}"
```

把這個鎖死，driver 換完再跑一次 `pytest tests/test_trace_schema.py`，就知道 schema 沒被破壞。

---

## 風險清單

| 風險 | 機率 | 緩解 |
|------|------|------|
| fiemap 在你的 fs 不支援 | 中 | `_lba_for` 已經有 try/except fallback |
| `O_DIRECT` 對齊要求嚴格 | 高 | 確保 buffer 是 block-aligned；用 `os.posix_memalign` 或 `mmap.mmap` |
| Driver oops 把整個 box 拖掛 | 中 | 在 VM 裡跑；常 snapshot |
| 真實 RMW count 跟 fallback 模型偏差大 | 高 (預期) | 這就是要做實驗的原因 |
| 改完跑不通，分不出哪步壞掉 | 高 | 嚴格照「漸進式驗證計畫」一步一停 |

---

## Summary checklist

- [ ] `IMRFIT_HDD_ROOT` 改 `/mnt/imr_hdd/wiki_corpus`
- [ ] `storage/tier_simulator.py:_lba_for` 走 fiemap
- [ ] `storage/tier_simulator.py:TierConfig.fsync_on_write = True`
- [ ] `imrsim/replay.py:_replay_kernel` 補滿 (現在是 stub)
- [ ] `scripts/02_setup_device.sh` mount IMR HDD
- [ ] 跑 `tests/test_trace_schema.py` 鎖 schema
- [ ] 照「漸進式驗證計畫」step 0→7 通過

完成這個 list 之後，`python run_experiment.py --scenario all` 應該還是同一行指令，只是底下打到的是真實的 IMR HDD。
