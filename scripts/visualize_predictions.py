from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio import denormalize_logmel, normalize_logmel
from config import TrainConfig
from data import TextMemmapAudioDataset
from model import AudioDiffusionGPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/ljspeech_bigvgan")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=".artifacts/local/prediction_viz.png")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{key: value for key, value in ckpt["config"].items() if key in cfg_fields})
    y_logmel, text = next(iter(TextMemmapAudioDataset(args.data_dir, cfg, seed=args.seed)))
    frames = min(args.frames or y_logmel.shape[0], y_logmel.shape[0])
    y_logmel = y_logmel[:frames].to(device)

    model = AudioDiffusionGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        hidden, text_mask = model.text_encoder([text], device)
        text_memory = model.text_proj(hidden.to(dtype=model.text_proj.weight.dtype))
        target_norm = normalize_logmel(y_logmel[None], cfg.mel_mean, cfg.mel_std)
        t = torch.full((1, frames, 1), 0.5, device=device)
        noise = torch.randn_like(target_norm)
        x_t = 0.5 * target_norm + 0.5 * noise
        v_target = noise - target_norm
        v_pred = model.predict_velocity(x_t, t, text_memory, text_mask)
        x0_pred = x_t - t * v_pred
        pred_logmel = denormalize_logmel(x0_pred.squeeze(0), cfg.mel_mean, cfg.mel_std)
        flow_mse = F.mse_loss(v_pred, v_target).item()

    target = y_logmel.float().cpu()
    pred = pred_logmel.float().cpu()
    flow_err = (v_pred - v_target).abs().squeeze(0).cpu()
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)

    im0 = axes[0].imshow(target.T, origin="lower", aspect="auto", interpolation="nearest")
    axes[0].set_title("target log-mel")
    axes[0].set_ylabel("mel bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.02)

    im1 = axes[1].imshow(pred.T, origin="lower", aspect="auto", interpolation="nearest")
    axes[1].set_title("flow x0 estimate at t=0.5")
    axes[1].set_ylabel("mel bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.02)

    im2 = axes[2].imshow(
        (pred - target).abs().T, origin="lower", aspect="auto", interpolation="nearest"
    )
    axes[2].set_title("absolute log-mel error")
    axes[2].set_ylabel("mel bin")
    fig.colorbar(im2, ax=axes[2], fraction=0.02)

    im3 = axes[3].imshow(flow_err.T, origin="lower", aspect="auto", interpolation="nearest")
    axes[3].set_title(f"velocity absolute error, MSE={flow_mse:.4f}")
    axes[3].set_xlabel("frame")
    axes[3].set_ylabel("mel bin")
    fig.colorbar(im3, ax=axes[3], fraction=0.02)

    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
