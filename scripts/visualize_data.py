from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import TrainConfig
from data import TextMemmapAudioDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/ljspeech_bigvgan")
    p.add_argument("--item", type=int, default=0)
    p.add_argument("--frames", type=int, default=512)
    p.add_argument("--out", default=".artifacts/local/ljspeech_data_viz.png")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(data_dir=args.data_dir, block_size=args.frames)
    mel, text = next(iter(TextMemmapAudioDataset(args.data_dir, cfg, seed=args.item)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    im = ax.imshow(mel.T.cpu().numpy(), origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(f"LJSpeech training crop: {tuple(mel.shape)}, text={text[:100]!r}")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
