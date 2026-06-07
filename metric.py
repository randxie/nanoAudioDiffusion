from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(
        self,
        out_dir: str,
        run_name: str,
        config: dict[str, Any],
        use_wandb: bool = True,
        wandb_url: str | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.f = self.metrics_path.open("a", buffering=1)
        self.wandb = None

        if use_wandb:
            try:
                import wandb

                wandb_dir = self.out_dir / "wandb"
                wandb_dir.mkdir(parents=True, exist_ok=True)
                if wandb_url:
                    os.environ["WANDB_BASE_URL"] = wandb_url
                default_mode = "online" if wandb_url else "offline"
                os.environ.setdefault("WANDB_MODE", default_mode)
                os.environ.setdefault("WANDB_DIR", str(wandb_dir))
                self.wandb = wandb.init(
                    project="nanoAudioLLM",
                    name=run_name,
                    config=config,
                    dir=str(wandb_dir),
                    mode=os.environ.get("WANDB_MODE", "offline"),
                    reinit="finish_previous",
                )
            except Exception as exc:
                print(f"wandb unavailable, continuing with local JSONL metrics: {exc}", flush=True)
                self.wandb = None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        row = {"step": step, "time": time.time(), **metrics}
        self.f.write(json.dumps(row, sort_keys=True) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def print_train(self, metrics: dict[str, Any], step: int) -> None:
        parts = [
            f"step {step}",
            f"loss {metrics['train/loss']:.4f}",
            f"flow {metrics['train/flow_loss']:.4f}",
            f"dur {metrics['train/duration_loss']:.4f}",
            f"dur_s {metrics['train/duration_pred_sec']:.2f}/{metrics['train/duration_target_sec']:.2f}",
            f"mb {metrics['train/micro_batch_size']}",
            f"accum {metrics['train/grad_accum_steps']}",
            f"eff_bs {metrics['train/effective_batch_size']}",
            f"step_s {metrics['time/step_sec']:.3f}",
            f"sps {metrics['perf/samples_per_sec']:.2f}",
            f"frames_s {metrics['perf/frames_per_sec']:.0f}",
            f"lr {metrics['optim/lr']:.2e}",
        ]
        if "train/epoch" in metrics:
            parts.insert(2, f"epoch {metrics['train/epoch']:.2f}")
        if "optim/grad_norm" in metrics:
            parts.append(f"grad {metrics['optim/grad_norm']:.3f}")
        if "cuda/max_memory_gb" in metrics:
            parts.append(f"mem_gb {metrics['cuda/max_memory_gb']:.2f}")
        print(" ".join(parts), flush=True)

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
        self.f.close()
