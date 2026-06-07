from __future__ import annotations

import json
import random
import re
import string
import wave
from pathlib import Path

import torch

from audio import denormalize_logmel
from data import PackedTextIndex

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_for_wer(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (wav.clamp(-1, 1).cpu().numpy() * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


class LightweightEvaluator:
    def __init__(self, cfg, device: torch.device):
        if cfg.eval_prompts == "overfit":
            if cfg.overfit_first_n_samples <= 0:
                raise ValueError("eval_prompts=overfit requires overfit_first_n_samples > 0")
            text_index = PackedTextIndex(cfg.data_dir)
            prompts = [text_index.text(idx).strip() for idx in range(cfg.overfit_first_n_samples)]
            prompts = [text for text in prompts if text]
        else:
            prompt_path = Path(cfg.eval_prompts)
            if not prompt_path.exists():
                raise FileNotFoundError(f"Missing eval prompts: {prompt_path}")
            prompts = [
                line.strip() for line in prompt_path.read_text().splitlines() if line.strip()
            ]
        if not prompts:
            raise ValueError(f"No eval prompts found for {cfg.eval_prompts}")
        if cfg.eval_num_samples < 0:
            raise ValueError("eval_num_samples must be >= 0")
        if cfg.eval_steps < 1:
            raise ValueError("eval_steps must be >= 1")
        self.cfg = cfg
        self.device = device
        self.whisper_device = cfg.eval_whisper_device or str(self.device)
        if cfg.eval_num_samples > 0 and len(prompts) > cfg.eval_num_samples:
            prompts = random.Random(cfg.seed).sample(prompts, cfg.eval_num_samples)
        self.prompts = prompts
        self.root = Path(cfg.out_dir) / "eval"
        self.vocoder = None
        self.whisper = None
        self.output_sample_rate = cfg.sample_rate

    def _load_vocoder(self):
        if self.vocoder is None:
            import bigvgan

            self.vocoder = (
                bigvgan.BigVGAN._from_pretrained(
                    model_id=self.cfg.eval_vocoder_model,
                    revision=None,
                    cache_dir=None,
                    force_download=False,
                    proxies=None,
                    resume_download=False,
                    local_files_only=False,
                    token=None,
                    use_cuda_kernel=False,
                )
                .to(self.device)
                .eval()
            )
            self.vocoder.remove_weight_norm()
            self.output_sample_rate = int(self.vocoder.h.sampling_rate)
        return self.vocoder

    def _load_whisper(self):
        if not self.cfg.eval_whisper_model:
            return None
        if self.whisper is None:
            import whisper

            self.whisper = whisper.load_model(
                self.cfg.eval_whisper_model, device=self.whisper_device
            )
        return self.whisper

    @torch.no_grad()
    def run(self, model, step: int) -> dict[str, float | str]:
        was_training = model.training
        model.eval()
        vocoder = self._load_vocoder()
        whisper_model = self._load_whisper()
        step_dir = self.root / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        refs_norm = []
        hyps_norm = []
        pred_seconds = []
        autocast = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and self.cfg.dtype == "bf16",
        )

        try:
            for idx, text in enumerate(self.prompts):
                hidden, text_mask = model.text_encoder([text], self.device)
                text_memory = model.text_proj(hidden.to(dtype=model.text_proj.weight.dtype))
                text_weights = text_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
                pooled = (hidden.float() * text_weights).sum(dim=1) / text_weights.sum(
                    dim=1
                ).clamp_min(1.0)
                predicted_seconds = (
                    model.duration_head(pooled.to(dtype=model.duration_head.weight.dtype))
                    .squeeze(-1)
                    .float()
                    .exp()
                    .item()
                )
                pred_seconds.append(predicted_seconds)
                seconds = max(0.5, min(12.0, predicted_seconds))
                frames = max(1, int(seconds * self.cfg.sample_rate / self.cfg.hop_length))

                x = torch.randn(1, frames, self.cfg.n_mels, device=self.device)
                time_grid = torch.linspace(1.0, 0.0, self.cfg.eval_steps + 1, device=self.device)
                cfg_text_memory = cfg_text_mask = None
                if self.cfg.eval_cfg_scale != 1.0:
                    cfg_hidden, cfg_text_mask = model.text_encoder(["", text], self.device)
                    cfg_text_memory = model.text_proj(
                        cfg_hidden.to(dtype=model.text_proj.weight.dtype)
                    )
                for i in range(self.cfg.eval_steps):
                    t = time_grid[i].view(1, 1, 1)
                    dt = (time_grid[i] - time_grid[i + 1]).float()
                    with autocast:
                        if self.cfg.eval_cfg_scale == 1.0:
                            velocity = model.predict_velocity(x, t, text_memory, text_mask)
                        else:
                            x_in = torch.cat((x, x), dim=0)
                            t_in = t.expand(2, -1, -1)
                            uncond_pred, cond_pred = model.predict_velocity(
                                x_in, t_in, cfg_text_memory, cfg_text_mask
                            ).chunk(2, dim=0)
                            velocity = uncond_pred + self.cfg.eval_cfg_scale * (
                                cond_pred - uncond_pred
                            )
                    x = x - velocity * dt

                logmel = denormalize_logmel(x.squeeze(0).cpu(), self.cfg.mel_mean, self.cfg.mel_std)
                wav = (
                    vocoder(logmel.transpose(0, 1).unsqueeze(0).to(self.device).float())
                    .squeeze()
                    .cpu()
                )
                wav_path = step_dir / f"{idx:06d}.wav"
                save_wav(wav_path, wav, self.output_sample_rate)

                hyp = ""
                wer = None
                if whisper_model is not None:
                    result = whisper_model.transcribe(
                        str(wav_path),
                        language="en",
                        task="transcribe",
                        fp16=self.whisper_device.startswith("cuda"),
                    )
                    import jiwer

                    hyp = " ".join(str(result["text"]).split())
                    refs_norm.append(normalize_for_wer(text))
                    hyps_norm.append(normalize_for_wer(hyp))
                    wer = jiwer.wer(refs_norm[-1], hyps_norm[-1])
                rows.append(
                    {
                        "idx": idx,
                        "audio": str(wav_path),
                        "reference": text,
                        "hypothesis": hyp,
                        "wer": wer,
                        "predicted_seconds": predicted_seconds,
                    }
                )
        finally:
            if self.whisper is not None:
                whisper_model = None
                self.whisper = None
                if self.whisper_device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if was_training:
                model.train()

        wer_path = step_dir / "wer.jsonl"
        with wer_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        metrics: dict[str, float | str] = {
            "eval/predicted_seconds": sum(pred_seconds) / max(1, len(pred_seconds)),
            "eval/num_samples": float(len(rows)),
            "eval/path": str(step_dir),
        }
        if refs_norm:
            import jiwer

            metrics["eval/wer"] = jiwer.wer(refs_norm, hyps_norm)
        return metrics
