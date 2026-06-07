import argparse
import os
import wave

import torch

from audio import denormalize_logmel
from config import TrainConfig
from model import AudioDiffusionGPT


def save_wav(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (wav.clamp(-1, 1).cpu().numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vocoder_model", default="nvidia/bigvgan_v2_22khz_80band_fmax8k_256x")
    args = p.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{key: value for key, value in ckpt["config"].items() if key in cfg_fields})
    model = AudioDiffusionGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    hidden, text_mask = model.text_encoder([args.text], device)
    text_memory = model.text_proj(hidden.to(dtype=model.text_proj.weight.dtype))
    text_weights = text_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    pooled = (hidden.float() * text_weights).sum(dim=1) / text_weights.sum(dim=1).clamp_min(1.0)
    predicted_seconds = (
        model.duration_head(pooled.to(dtype=model.duration_head.weight.dtype))
        .squeeze(-1)
        .float()
        .exp()
        .item()
    )
    seconds = args.seconds if args.seconds > 0 else max(0.5, min(12.0, predicted_seconds))
    frames = max(1, int(seconds * cfg.sample_rate / cfg.hop_length))

    x = torch.randn(1, frames, cfg.n_mels, device=device)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    time_grid = torch.linspace(1.0, 0.0, args.steps + 1, device=device)
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and cfg.dtype == "bf16",
    )
    cfg_text_memory = cfg_text_mask = None
    if args.cfg_scale != 1.0:
        cfg_hidden, cfg_text_mask = model.text_encoder(["", args.text], device)
        cfg_text_memory = model.text_proj(cfg_hidden.to(dtype=model.text_proj.weight.dtype))
    for i in range(args.steps):
        t = time_grid[i].view(1, 1, 1)
        dt = (time_grid[i] - time_grid[i + 1]).float()
        with autocast:
            if args.cfg_scale == 1.0:
                velocity = model.predict_velocity(x, t, text_memory, text_mask)
            else:
                x_in = torch.cat((x, x), dim=0)
                t_in = t.expand(2, -1, -1)
                uncond_pred, cond_pred = model.predict_velocity(
                    x_in, t_in, cfg_text_memory, cfg_text_mask
                ).chunk(2, dim=0)
                velocity = uncond_pred + args.cfg_scale * (cond_pred - uncond_pred)
        x = x - velocity * dt

    logmel = denormalize_logmel(x.squeeze(0).cpu(), cfg.mel_mean, cfg.mel_std)
    import bigvgan

    vocoder = (
        bigvgan.BigVGAN._from_pretrained(
            model_id=args.vocoder_model,
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
            use_cuda_kernel=False,
        )
        .to(device)
        .eval()
    )
    vocoder.remove_weight_norm()
    output_sample_rate = int(vocoder.h.sampling_rate)
    wav = vocoder(logmel.transpose(0, 1).unsqueeze(0).to(device).float()).squeeze().cpu()
    save_wav(args.out, wav, output_sample_rate)
    print(
        f"wrote {args.out} seconds={seconds:.2f} predicted_seconds={predicted_seconds:.2f} "
        f"steps={args.steps} cfg_scale={args.cfg_scale:.2f}"
    )


if __name__ == "__main__":
    main()
