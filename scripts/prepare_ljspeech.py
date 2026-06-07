from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio import logmel_from_wav


def audio_index_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("utt_id", pa.string()),
            ("source_shard", pa.string()),
            ("source_member", pa.string()),
            ("audio_ext", pa.string()),
            ("pack_file", pa.string()),
            ("byte_offset", pa.int64()),
            ("byte_length", pa.int64()),
            ("sha256", pa.string()),
            ("duration_sec", pa.float32()),
            ("sample_rate", pa.int32()),
            ("num_frames", pa.int32()),
            ("text", pa.string()),
            ("speaker", pa.string()),
            ("language", pa.string()),
            ("metadata_json", pa.string()),
            ("decode_error", pa.string()),
        ]
    )


class MelShardWriter:
    def __init__(self, root: Path, args: argparse.Namespace):
        self.root = root / "mel_shards"
        self.root.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.max_frames = args.mel_shard_frames
        self.shard_idx = 0
        self.shard_dir: Path | None = None
        self.logmel_f = None
        self.lengths: list[int] = []
        self.offsets: list[int] = []
        self.num_frames = 0

    def _open_next(self) -> None:
        self.finalize()
        shard_dir = self.root / f"shard_{self.shard_idx:03d}"
        self.shard_idx += 1
        shard_dir.mkdir(parents=True)
        self.shard_dir = shard_dir
        self.logmel_f = (shard_dir / "logmel.float16.memmap").open("ab")
        self.lengths = []
        self.offsets = []
        self.num_frames = 0

    def append(self, logmel: np.ndarray) -> None:
        frames = int(logmel.shape[0])
        if frames <= 0:
            return
        if self.shard_dir is None or (
            self.num_frames and self.num_frames + frames > self.max_frames
        ):
            self._open_next()
        assert self.logmel_f is not None
        self.offsets.append(self.num_frames)
        self.lengths.append(frames)
        np.ascontiguousarray(logmel, dtype=np.float16).tofile(self.logmel_f)
        self.num_frames += frames

    def finalize(self) -> None:
        if self.shard_dir is None:
            return
        assert self.logmel_f is not None
        self.logmel_f.close()
        np.save(self.shard_dir / "lengths.int32.npy", np.asarray(self.lengths, dtype=np.int32))
        np.save(self.shard_dir / "offsets.int64.npy", np.asarray(self.offsets, dtype=np.int64))
        meta = {
            "sample_rate": self.args.sample_rate,
            "n_fft": self.args.n_fft,
            "hop_length": self.args.hop_length,
            "win_length": self.args.win_length,
            "n_mels": self.args.n_mels,
            "f_min": self.args.f_min,
            "f_max": self.args.f_max,
            "num_frames": self.num_frames,
            "num_utts": len(self.lengths),
        }
        with (self.shard_dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        self.shard_dir = None
        self.logmel_f = None


def write_parquet_part(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows, schema=audio_index_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def combine_parquet_parts(parts_dir: Path, index_path: Path) -> int:
    import pyarrow.parquet as pq

    parts = sorted(parts_dir.glob("*.parquet"))
    rows = 0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(index_path, schema=audio_index_schema()) as writer:
        for part in parts:
            parquet_file = pq.ParquetFile(part)
            rows += parquet_file.metadata.num_rows
            for batch in parquet_file.iter_batches(batch_size=65536):
                writer.write_batch(batch)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--out_dir", default="data/ljspeech_bigvgan")
    p.add_argument("--download", action="store_true")
    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--f_min", type=float, default=0.0)
    p.add_argument("--f_max", type=float, default=8000.0)
    p.add_argument("--mel_shard_frames", type=int, default=2_000_000)
    p.add_argument("--rows_per_part", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if (out_dir / "mel_shards").exists():
        raise SystemExit(f"{out_dir}/mel_shards already exists; remove it before rebuilding.")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_index_schema()

    dataset = torchaudio.datasets.LJSPEECH(args.root, download=args.download)
    resamplers = {}
    mel_writer = MelShardWriter(out_dir, args)
    parts_dir = out_dir / "metadata" / "audio_index_parts"
    rows = []
    value_sum = 0.0
    value_sq_sum = 0.0
    value_count = 0

    try:
        for idx in range(len(dataset)):
            wav, src_sr, transcript, normalized_transcript = dataset[idx]
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
            if src_sr != args.sample_rate:
                if src_sr not in resamplers:
                    resamplers[src_sr] = torchaudio.transforms.Resample(src_sr, args.sample_rate)
                wav = resamplers[src_sr](wav)
            wav = wav.float()
            logmel = logmel_from_wav(
                wav,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                n_mels=args.n_mels,
                f_min=args.f_min,
                f_max=args.f_max,
            )
            mel_writer.append(logmel.cpu().numpy().astype(np.float16, copy=False))
            values = logmel.double()
            value_sum += values.sum().item()
            value_sq_sum += values.square().sum().item()
            value_count += values.numel()
            rows.append(
                {
                    "utt_id": f"ljspeech-{idx:05d}",
                    "source_shard": "LJSpeech-1.1",
                    "source_member": f"{idx:05d}",
                    "audio_ext": "wav",
                    "pack_file": None,
                    "byte_offset": None,
                    "byte_length": None,
                    "sha256": None,
                    "duration_sec": float(wav.numel()) / float(args.sample_rate),
                    "sample_rate": args.sample_rate,
                    "num_frames": int(logmel.shape[0]),
                    "text": normalized_transcript or transcript,
                    "speaker": "LJ",
                    "language": "en",
                    "metadata_json": json.dumps({"transcript": transcript}, ensure_ascii=False),
                    "decode_error": None,
                }
            )
            if len(rows) >= args.rows_per_part:
                write_parquet_part(rows, parts_dir / f"part_{idx:06d}.parquet")
                rows = []
    finally:
        mel_writer.finalize()

    if rows:
        write_parquet_part(rows, parts_dir / "part_final.parquet")
    row_count = combine_parquet_parts(parts_dir, out_dir / "metadata" / "audio_index.parquet")
    mean = value_sum / max(1, value_count)
    var = value_sq_sum / max(1, value_count) - mean * mean
    stats = {
        "mel_mean": mean,
        "mel_std": var**0.5,
        "num_values": value_count,
        "audio_rows": row_count,
    }
    with (out_dir / "metadata" / "mel_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {row_count} rows to {out_dir}")
    print(f"mel_mean: {stats['mel_mean']:.6f}")
    print(f"mel_std: {stats['mel_std']:.6f}")


if __name__ == "__main__":
    main()
