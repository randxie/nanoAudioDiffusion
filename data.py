import json
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class _MemmapShard:
    def __init__(self, data_dir: Path):
        with open(data_dir / "meta.json") as f:
            self.meta = json.load(f)
        shape = (self.meta["num_frames"], self.meta["n_mels"])
        self.logmel = np.memmap(
            data_dir / "logmel.float16.memmap", dtype=np.float16, mode="r", shape=shape
        )
        self.lengths = np.load(data_dir / "lengths.int32.npy")
        self.offsets = np.load(data_dir / "offsets.int64.npy")
        self.num_utts = len(self.lengths)


def _find_memmap_shards(data_dir: str):
    root = Path(data_dir)
    if (root / "meta.json").exists():
        return [root]
    mel_root = root / "mel_shards"
    if mel_root.exists():
        shards = sorted(path for path in mel_root.glob("shard_*") if (path / "meta.json").exists())
        if shards:
            return shards
    raise FileNotFoundError(f"No memmap shard found under {data_dir}")


class MemmapAudioDataset(IterableDataset):
    def __init__(
        self, data_dir: str, block_size: int, rank: int = 0, world_size: int = 1, seed: int = 1234
    ):
        self.data_dir = data_dir
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shards = [_MemmapShard(path) for path in _find_memmap_shards(data_dir)]
        self.cum_utts = np.cumsum([shard.num_utts for shard in self.shards])
        self.meta = self.shards[0].meta


class TextMemmapAudioDataset(MemmapAudioDataset):
    def __init__(self, data_dir: str, cfg, rank: int = 0, world_size: int = 1, seed: int = 1234):
        super().__init__(data_dir, cfg.block_size, rank, world_size, seed)
        self.cfg = cfg
        self._validate_mel_config()
        self.text_index = PackedTextIndex(data_dir)
        self.has_full_eligible = any(
            np.any(shard.lengths <= self.block_size) for shard in self.shards
        )

    def _validate_mel_config(self):
        expected = {
            "sample_rate": self.cfg.sample_rate,
            "n_fft": self.cfg.n_fft,
            "hop_length": self.cfg.hop_length,
            "win_length": self.cfg.win_length,
            "n_mels": self.cfg.n_mels,
            "f_min": self.cfg.mel_f_min,
            "f_max": self.cfg.mel_f_max,
        }
        meta = self.shards[0].meta
        missing = [key for key in expected if key not in meta]
        if missing:
            raise ValueError(
                f"Dataset mel metadata is missing {missing}; regenerate mels for the current config."
            )
        mismatched = []
        for key, value in expected.items():
            actual = meta[key]
            if isinstance(value, float):
                ok = abs(float(actual) - value) < 1e-6
            else:
                ok = int(actual) == int(value)
            if not ok:
                mismatched.append(f"{key}: dataset={actual} config={value}")
        if mismatched:
            raise ValueError(
                "Dataset mel config mismatch; regenerate mels. " + "; ".join(mismatched)
            )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + self.rank * 1000 + worker_id)
        max_silence_frames = int(
            round(self.cfg.max_silence_sec * self.cfg.sample_rate / self.cfg.hop_length)
        )
        overfit_count = min(int(self.cfg.overfit_first_n_samples), int(self.cum_utts[-1]))
        if overfit_count > 0:
            worker_count = worker.num_workers if worker is not None else 1
            stride = max(1, self.world_size * worker_count)
            offset = self.rank * worker_count + worker_id
            overfit_ids = list(range(offset, overfit_count, stride))
            if not overfit_ids:
                overfit_ids = [offset % overfit_count]
            idx = 0
            while True:
                global_utt = overfit_ids[idx % len(overfit_ids)]
                idx += 1
                shard_idx = bisect_right(self.cum_utts, global_utt)
                prev = int(self.cum_utts[shard_idx - 1]) if shard_idx else 0
                shard = self.shards[shard_idx]
                utt = global_utt - prev
                frames = min(int(shard.lengths[utt]), self.block_size)
                if frames < 1:
                    continue
                start = shard.offsets[utt]
                y_logmel = torch.from_numpy(np.array(shard.logmel[start : start + frames]))
                if max_silence_frames > 0 and rng.random() < self.cfg.silence_prob:
                    max_add = min(max_silence_frames, self.block_size - y_logmel.shape[0])
                    if max_add > 0:
                        add = int(rng.integers(max_add + 1))
                        if add > 0:
                            silence = torch.full(
                                (add, self.cfg.n_mels), math.log(1e-5), dtype=y_logmel.dtype
                            )
                            y_logmel = torch.cat([y_logmel, silence])
                yield y_logmel, self.text_index.text(global_utt)

        while True:
            global_utt = int(rng.integers(int(self.cum_utts[-1])))
            shard_idx = bisect_right(self.cum_utts, global_utt)
            prev = int(self.cum_utts[shard_idx - 1]) if shard_idx else 0
            shard = self.shards[shard_idx]
            utt = global_utt - prev
            frames = int(shard.lengths[utt])
            if frames < 1:
                continue
            if frames > self.block_size:
                if self.has_full_eligible:
                    continue
                start = shard.offsets[utt] + rng.integers(frames - self.block_size + 1)
                frames = self.block_size
            else:
                start = shard.offsets[utt]
            y_logmel = torch.from_numpy(np.array(shard.logmel[start : start + frames]))
            if max_silence_frames > 0 and rng.random() < self.cfg.silence_prob:
                max_add = min(max_silence_frames, self.block_size - y_logmel.shape[0])
                if max_add > 0:
                    add = int(rng.integers(max_add + 1))
                    if add > 0:
                        silence = torch.full(
                            (add, self.cfg.n_mels), math.log(1e-5), dtype=y_logmel.dtype
                        )
                        y_logmel = torch.cat([y_logmel, silence])
            text = self.text_index.text(global_utt)
            yield y_logmel, text


class PackedTextIndex:
    def __init__(
        self, data_dir: str, index_name: str = "metadata/audio_index.parquet", cache_size: int = 8
    ):
        import pyarrow.parquet as pq

        self.parquet = pq.ParquetFile(Path(data_dir) / index_name)
        self.row_group_counts = [
            self.parquet.metadata.row_group(i).num_rows
            for i in range(self.parquet.metadata.num_row_groups)
        ]
        self.cum_rows = np.cumsum(self.row_group_counts)
        self.cache_size = cache_size
        self.cache: dict[int, list[str | None]] = {}

    def text(self, idx: int) -> str:
        row_group = bisect_right(self.cum_rows, idx)
        prev = int(self.cum_rows[row_group - 1]) if row_group else 0
        if row_group not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            rows = (
                self.parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
            )
            self.cache[row_group] = rows
        text = self.cache[row_group][idx - prev]
        return text or ""


def make_dataset(cfg, rank: int, world_size: int = 1):
    if not cfg.data_dir:
        raise ValueError("data_dir is required")
    return TextMemmapAudioDataset(cfg.data_dir, cfg, rank, world_size, cfg.seed)


def audio_collate(batch):
    batch_size = len(batch)
    max_len = max(item[0].shape[0] for item in batch)
    n_mels = batch[0][0].shape[1]
    mel = torch.zeros(batch_size, max_len, n_mels, dtype=batch[0][0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    texts = []
    for idx, (mel_i, text) in enumerate(batch):
        length = mel_i.shape[0]
        mel[idx, :length] = mel_i
        mask[idx, :length] = True
        texts.append(text)
    return mel, mask, texts
