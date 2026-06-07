from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    train_rows = []
    wer_by_step = {}
    with Path(args.metrics).open() as f:
        for line in f:
            row = json.loads(line)
            if "train/loss" in row:
                train_rows.append(row)
            if "eval/wer" in row:
                wer_by_step[int(row["step"])] = float(row["eval/wer"])
    if not train_rows:
        raise ValueError(f"No train/loss rows found in {args.metrics}")

    steps = [row["step"] for row in train_rows]
    loss = [row["train/loss"] for row in train_rows]
    flow = [row["train/flow_loss"] for row in train_rows]
    duration = [row["train/duration_loss"] for row in train_rows]
    wer_steps = sorted(wer_by_step)
    wer = [wer_by_step[step] for step in wer_steps]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, loss, label="total", linewidth=1.5)
    axes[0].plot(steps, flow, label="flow", linewidth=1.2)
    axes[0].plot(steps, duration, label="duration", linewidth=1.0)
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    if wer_steps:
        axes[1].plot(
            wer_steps, wer, color="tab:red", marker="o", label="wer", linewidth=1.5, markersize=3
        )
    else:
        axes[1].text(
            0.5, 0.5, "no eval/wer rows", ha="center", va="center", transform=axes[1].transAxes
        )
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("wer")
    axes[1].set_ylim(0.2, 1.2)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
