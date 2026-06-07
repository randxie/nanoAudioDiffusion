from dataclasses import dataclass, fields

from audio import BIGVGAN_LOGMEL_MEAN, BIGVGAN_LOGMEL_STD


@dataclass
class TrainConfig:
    # Mel/audio sequence shape.
    block_size: int = 1024
    n_mels: int = 80

    # Frozen text encoder and conditioning.
    text_encoder_name: str = "google/mt5-small"
    text_encoder_trainable: bool = False
    text_embedding_trainable: bool = False
    text_condition_dropout: float = 0.0
    max_text_length: int = 256

    # Transformer mel flow decoder.
    decoder_d_model: int = 512
    decoder_n_layer: int = 8
    decoder_n_head: int = 8
    decoder_dropout: float = 0.0
    rope_theta: float = 10000.0

    # Duration prediction and tail-silence augmentation.
    lambda_duration: float = 0.1
    silence_prob: float = 0.5
    max_silence_sec: float = 0.5

    # Flow-matching timestep sampling.
    diffusion_train_steps: int = 1000
    timestep_logit_mean: float = -0.8
    timestep_logit_std: float = 0.8

    # Log-mel normalization.
    mel_mean: float = BIGVGAN_LOGMEL_MEAN
    mel_std: float = BIGVGAN_LOGMEL_STD

    # Batch size and optimizer schedule.
    batch_size: int = 2
    grad_accum_steps: int = 16
    effective_batch_size: int = 0
    lr: float = 5e-4
    min_lr: float = 0.0
    cosine_lr: bool = False
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 100000
    lr_schedule_steps: int = 0
    max_epochs: int = 0

    # Logging and checkpoint cadence.
    log_interval: int = 10
    val_interval: int = 1000
    save_interval: int = 1000

    # Lightweight in-loop eval.
    eval_interval: int = 0
    eval_prompts: str = "eval/libritts_samples.txt"
    eval_num_samples: int = 4
    eval_steps: int = 16
    eval_cfg_scale: float = 2.0
    eval_whisper_model: str = "base.en"
    eval_whisper_device: str = ""
    eval_vocoder_model: str = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"

    # Data loading and run control.
    prefetch_factor: int = 2
    max_train_seconds: float = 0.0
    dtype: str = "bf16"
    data_dir: str = ""
    overfit_first_n_samples: int = 0
    out_dir: str = "out"
    compile: bool = False
    seed: int = 1234
    num_workers: int = 4

    # BigVGAN-compatible mel extraction.
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_f_min: float = 0.0
    mel_f_max: float = 8000.0


def load_config(path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if not path:
        return cfg
    names = {f.name: f.type for f in fields(cfg)}
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = [x.strip() for x in line.split(":", 1)]
            if key not in names:
                continue
            old = getattr(cfg, key)
            if isinstance(old, bool):
                value = value.lower() in {"1", "true", "yes", "on"}
            elif isinstance(old, int):
                value = int(value)
            elif isinstance(old, float):
                value = float(value)
            setattr(cfg, key, value)
    return cfg
