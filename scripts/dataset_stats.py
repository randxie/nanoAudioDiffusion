from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def find_shards(data_dir: Path) -> list[Path]:
    if (data_dir / "meta.json").exists():
        return [data_dir]
    shards = sorted(
        path for path in (data_dir / "mel_shards").glob("shard_*") if (path / "meta.json").exists()
    )
    if not shards:
        raise FileNotFoundError(f"No mel shards found under {data_dir}")
    return shards


def mel_stats(shards: list[Path], chunk_frames: int) -> tuple[float, float, int, int, int]:
    total = 0.0
    total_sq = 0.0
    total_values = 0
    total_frames = 0
    total_utts = 0
    for shard in shards:
        with (shard / "meta.json").open() as f:
            meta = json.load(f)
        num_frames = int(meta["num_frames"])
        n_mels = int(meta["n_mels"])
        logmel = np.memmap(
            shard / "logmel.float16.memmap", dtype=np.float16, mode="r", shape=(num_frames, n_mels)
        )
        lengths = np.load(shard / "lengths.int32.npy")
        total_frames += num_frames
        total_utts += len(lengths)
        for start in range(0, num_frames, chunk_frames):
            values = logmel[start : start + chunk_frames].astype(np.float64)
            total += values.sum().item()
            total_sq += np.square(values).sum().item()
            total_values += values.size
    mean = total / max(1, total_values)
    var = total_sq / max(1, total_values) - mean * mean
    return mean, max(0.0, var) ** 0.5, total_values, total_frames, total_utts


def text_stats(
    data_dir: Path, text_encoder_name: str, max_text_length: int, batch_size: int
) -> dict[str, float | int | str]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    parquet = pq.ParquetFile(data_dir / "metadata" / "audio_index.parquet")
    token_total = 0
    token_max = 0
    truncated = 0
    char_total = 0
    samples = 0
    duration_total = 0.0
    frame_total = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["text", "duration_sec", "num_frames"]
    ):
        texts = [text or "" for text in batch.column("text").to_pylist()]
        durations = batch.column("duration_sec").to_pylist()
        frames = batch.column("num_frames").to_pylist()
        tokenized = tokenizer(texts, add_special_tokens=True, truncation=False)
        for text, input_ids, duration_sec, num_frames in zip(
            texts, tokenized["input_ids"], durations, frames, strict=True
        ):
            token_count = len(input_ids)
            token_total += token_count
            token_max = max(token_max, token_count)
            truncated += int(token_count > max_text_length)
            char_total += len(text)
            duration_total += float(duration_sec)
            frame_total += int(num_frames)
            samples += 1
    return {
        "text_encoder_name": text_encoder_name,
        "max_text_length": max_text_length,
        "num_samples": samples,
        "avg_tokens": token_total / max(1, samples),
        "max_tokens": token_max,
        "num_over_max_text_length": truncated,
        "avg_text_chars": char_total / max(1, samples),
        "avg_duration_sec": duration_total / max(1, samples),
        "total_duration_hours": duration_total / 3600.0,
        "avg_frames": frame_total / max(1, samples),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ljspeech.yaml")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--chunk_frames", type=int, default=65536)
    p.add_argument("--token_batch_size", type=int, default=256)
    p.add_argument("--text_encoder_name", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir_arg = args.data_dir or cfg.data_dir
    if not data_dir_arg:
        raise ValueError("data_dir is required")
    data_dir = Path(data_dir_arg)
    text_encoder_name = args.text_encoder_name or cfg.text_encoder_name
    shards = find_shards(data_dir)
    mean, std, num_values, total_frames, total_utts = mel_stats(shards, args.chunk_frames)
    stats = {
        "data_dir": str(data_dir),
        "num_shards": len(shards),
        "num_mel_utts": total_utts,
        "total_frames": total_frames,
        "mel_mean": mean,
        "mel_std": std,
        "mel_num_values": num_values,
        **text_stats(data_dir, text_encoder_name, cfg.max_text_length, args.token_batch_size),
    }
    out = Path(args.out) if args.out else data_dir / "metadata" / "dataset_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
