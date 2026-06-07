from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio import denormalize_logmel
from config import TrainConfig
from evaluator import normalize_for_wer, save_wav
from model import AudioDiffusionGPT


def load_vocoder(model_id: str, device: torch.device):
    import bigvgan

    vocoder = (
        bigvgan.BigVGAN._from_pretrained(
            model_id=model_id,
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
    return vocoder


@torch.no_grad()
def sample_wav(model, cfg, vocoder, text: str, steps: int, cfg_scale: float, device: torch.device):
    hidden, text_mask = model.text_encoder([text], device)
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
    seconds = max(0.5, min(12.0, predicted_seconds))
    frames = max(1, int(seconds * cfg.sample_rate / cfg.hop_length))

    x = torch.randn(1, frames, cfg.n_mels, device=device)
    time_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and cfg.dtype == "bf16",
    )
    cfg_text_memory = cfg_text_mask = None
    if cfg_scale != 1.0:
        cfg_hidden, cfg_text_mask = model.text_encoder(["", text], device)
        cfg_text_memory = model.text_proj(cfg_hidden.to(dtype=model.text_proj.weight.dtype))
    for i in range(steps):
        t = time_grid[i].view(1, 1, 1)
        dt = (time_grid[i] - time_grid[i + 1]).float()
        with autocast:
            if cfg_scale == 1.0:
                velocity = model.predict_velocity(x, t, text_memory, text_mask)
            else:
                x_in = torch.cat((x, x), dim=0)
                t_in = t.expand(2, -1, -1)
                uncond_pred, cond_pred = model.predict_velocity(
                    x_in, t_in, cfg_text_memory, cfg_text_mask
                ).chunk(2, dim=0)
                velocity = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        x = x - velocity * dt

    logmel = denormalize_logmel(x.squeeze(0).cpu(), cfg.mel_mean, cfg.mel_std)
    wav = vocoder(logmel.transpose(0, 1).unsqueeze(0).to(device).float()).squeeze().cpu()
    return wav, predicted_seconds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--whisper_model", default="base.en")
    p.add_argument("--whisper_device", default="cuda")
    p.add_argument("--vocoder_model", default="nvidia/bigvgan_v2_22khz_80band_fmax8k_256x")
    args = p.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    whisper_device = (
        args.whisper_device if torch.cuda.is_available() or args.whisper_device == "cpu" else "cpu"
    )
    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if args.limit > 0:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{key: value for key, value in ckpt["config"].items() if key in cfg_fields})
    model = AudioDiffusionGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocoder = load_vocoder(args.vocoder_model, device)
    sample_rate = int(vocoder.h.sampling_rate)

    whisper_model = None
    if args.whisper_model:
        import whisper

        whisper_model = whisper.load_model(args.whisper_model, device=whisper_device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    refs_norm = []
    hyps_norm = []
    for idx, text in enumerate(prompts):
        wav, predicted_seconds = sample_wav(
            model, cfg, vocoder, text, args.steps, args.cfg_scale, device
        )
        wav_path = out_dir / f"{idx:06d}.wav"
        save_wav(wav_path, wav, sample_rate)
        hyp = ""
        wer = None
        if whisper_model is not None:
            import jiwer

            result = whisper_model.transcribe(
                str(wav_path),
                language="en",
                task="transcribe",
                fp16=whisper_device.startswith("cuda"),
            )
            hyp = " ".join(str(result["text"]).split())
            refs_norm.append(normalize_for_wer(text))
            hyps_norm.append(normalize_for_wer(hyp))
            wer = jiwer.wer(refs_norm[-1], hyps_norm[-1])
        row = {
            "idx": idx,
            "audio": str(wav_path),
            "reference": text,
            "hypothesis": hyp,
            "wer": wer,
            "predicted_seconds": predicted_seconds,
        }
        rows.append(row)
        print(f"{idx:06d} seconds={predicted_seconds:.2f} wer={wer} wav={wav_path}", flush=True)

    summary = {
        "count": len(rows),
        "ckpt": args.ckpt,
        "prompts": args.prompts,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "whisper_model": args.whisper_model,
    }
    if refs_norm:
        import jiwer

        summary["wer"] = jiwer.wer(refs_norm, hyps_norm)
    with (out_dir / "wer.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.write(json.dumps({"idx": "summary", **summary}, sort_keys=True) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if whisper_model is not None:
        del whisper_model
        if whisper_device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"wrote {out_dir / 'wer.jsonl'}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
