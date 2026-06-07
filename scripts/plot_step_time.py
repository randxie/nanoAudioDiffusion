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
    p.add_argument("--out", default=None)
    p.add_argument("--skip_first", type=int, default=1)
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    rows = []
    with metrics_path.open() as f:
        for line in f:
            row = json.loads(line)
            if "time/step_sec" in row:
                rows.append(row)
    if not rows:
        raise ValueError(f"No time/step_sec rows found in {metrics_path}")

    steps = [row["step"] for row in rows]
    step_sec = [row["time/step_sec"] for row in rows]
    kept = step_sec[args.skip_first :] if len(step_sec) > args.skip_first else step_sec
    mean_step_sec = sum(kept) / len(kept)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, step_sec, marker="o", linewidth=1.5)
    ax.axhline(
        mean_step_sec,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"mean {mean_step_sec:.3f}s",
    )
    ax.set_title("Training Step Time")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else metrics_path.with_name("step_time.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
