from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from model import AudioDiffusionGPT


def format_flops(value: int) -> str:
    return f"{value:,} ({value / 1e9:.2f}G)"


def measure_flop_breakdown(
    model: AudioDiffusionGPT, batch_size: int, text_len: int, seq_len: int, device: torch.device
):
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    input_ids = torch.ones(batch_size, text_len, dtype=torch.long, device=device)
    attention_mask = torch.ones(batch_size, text_len, dtype=torch.long, device=device)
    x_t = torch.randn(batch_size, seq_len, model.cfg.n_mels, device=device)
    t = torch.full((batch_size, 1, 1), 0.5, device=device)
    text_memory = torch.randn(batch_size, text_len, model.cfg.decoder_d_model, device=device)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool, device=device)

    with torch.no_grad(), FlopCounterMode(display=False) as mode:
        model.text_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
    encoder_flops = mode.get_total_flops()

    with torch.no_grad(), FlopCounterMode(display=False) as mode:
        model.flow(x_t, t, text_memory, text_mask)
    decoder_flops = mode.get_total_flops()

    if was_training:
        model.train()
    return {
        "text_encoder_forward_flops": encoder_flops,
        "mel_decoder_forward_flops": decoder_flops,
        "encoder_decoder_forward_flops": encoder_flops + decoder_flops,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ljspeech.yaml")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--text_len", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--flops", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--text_encoder_name", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.text_encoder_name is not None:
        cfg.text_encoder_name = args.text_encoder_name
    device = torch.device(args.device)
    model = AudioDiffusionGPT(cfg).to(device)
    summary = model.parameter_summary()

    print(f"config: {args.config}")
    for key, value in summary.items():
        print(f"{key}: {value:,} ({value / 1e6:.2f}M)")

    if args.flops:
        breakdown = measure_flop_breakdown(
            model, args.batch_size, args.text_len, args.seq_len, device
        )
        for key, value in breakdown.items():
            print(f"{key}: {format_flops(value)}")


if __name__ == "__main__":
    main()
