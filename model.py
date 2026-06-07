from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio import normalize_logmel


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, mask: torch.Tensor | None = None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).unbind(dim=2)
        q, k, v = [y.transpose(1, 2) for y in (q, k, v)]
        attn_mask = mask[:, None, None, :].bool() if mask is not None else None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, d))


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, rope_theta: float):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None
    ):
        b, t, d = x.shape
        s = memory.shape[1]
        q = self.q(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv(memory).view(b, s, 2, self.n_head, self.head_dim).unbind(dim=2)
        k, v = [y.transpose(1, 2) for y in (k, v)]
        q_pos = torch.arange(t, device=q.device, dtype=torch.float32)
        k_pos = torch.arange(s, device=k.device, dtype=torch.float32)
        inv_freq = self.rope_inv_freq.to(device=q.device)
        q_freqs = torch.outer(q_pos, inv_freq)
        k_freqs = torch.outer(k_pos, inv_freq)
        q_cos = q_freqs.cos()[None, None]
        q_sin = q_freqs.sin()[None, None]
        k_cos = k_freqs.cos()[None, None]
        k_sin = k_freqs.sin()[None, None]
        q_dtype = q.dtype
        k_dtype = k.dtype
        q_float = q.float()
        k_float = k.float()
        q1, q2 = q_float[..., 0::2], q_float[..., 1::2]
        k1, k2 = k_float[..., 0::2], k_float[..., 1::2]
        q = (
            torch.stack((q1 * q_cos - q2 * q_sin, q1 * q_sin + q2 * q_cos), dim=-1)
            .flatten(-2)
            .to(q_dtype)
        )
        k = (
            torch.stack((k1 * k_cos - k2 * k_sin, k1 * k_sin + k2 * k_cos), dim=-1)
            .flatten(-2)
            .to(k_dtype)
        )
        attn_mask = None
        if memory_mask is not None:
            attn_mask = memory_mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 8 * d_model, bias=False)
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(a) * b)


class MelTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, rope_theta: float):
        super().__init__()
        self.self_norm = RMSNorm(d_model)
        self.self_attn = SelfAttention(d_model, n_head, dropout)
        self.cross_norm = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_head, dropout, rope_theta)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        mel_mask: torch.Tensor | None,
    ):
        x = x + self.self_attn(self.self_norm(x), mel_mask)
        x = x + self.cross_attn(self.cross_norm(x), memory, memory_mask)
        return x + self.mlp(self.mlp_norm(x))


class MelTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(cfg.n_mels, cfg.decoder_d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(1, cfg.decoder_d_model),
            nn.SiLU(),
            nn.Linear(cfg.decoder_d_model, cfg.decoder_d_model),
        )
        self.blocks = nn.ModuleList(
            MelTransformerBlock(
                cfg.decoder_d_model, cfg.decoder_n_head, cfg.decoder_dropout, cfg.rope_theta
            )
            for _ in range(cfg.decoder_n_layer)
        )
        self.norm = RMSNorm(cfg.decoder_d_model)
        self.out_proj = nn.Linear(cfg.decoder_d_model, cfg.n_mels)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_memory: torch.Tensor,
        text_attention_mask: torch.Tensor | None,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.in_proj(x_t) + self.t_proj(t.to(dtype=x_t.dtype))
        for block in self.blocks:
            h = block(h, text_memory, text_attention_mask, mel_mask)
        return self.out_proj(self.norm(h))


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_text_length = cfg.max_text_length
        self.trainable = cfg.text_encoder_trainable

        try:
            from transformers import (
                AutoConfig,
                AutoModel,
                AutoTokenizer,
                MT5EncoderModel,
                T5EncoderModel,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the text encoder. Install with `.venv/bin/python -m pip install -e .`."
            ) from exc

        torch_dtype = torch.bfloat16 if cfg.dtype == "bf16" else None
        try:
            encoder_config = AutoConfig.from_pretrained(cfg.text_encoder_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.text_encoder_name, fix_mistral_regex=True
            )
            if encoder_config.model_type == "mt5":
                self.model = MT5EncoderModel.from_pretrained(
                    cfg.text_encoder_name, torch_dtype=torch_dtype
                )
            elif encoder_config.model_type in {"t5", "byt5"}:
                self.model = T5EncoderModel.from_pretrained(
                    cfg.text_encoder_name, torch_dtype=torch_dtype
                )
            else:
                self.model = AutoModel.from_pretrained(
                    cfg.text_encoder_name, torch_dtype=torch_dtype, attn_implementation="sdpa"
                )
        except Exception as exc:
            raise RuntimeError(f"Could not load text encoder {cfg.text_encoder_name!r}.") from exc
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.model.parameters():
            param.requires_grad = cfg.text_encoder_trainable
        for param in self.model.get_input_embeddings().parameters():
            param.requires_grad = cfg.text_embedding_trainable and cfg.text_encoder_trainable
        if not cfg.text_encoder_trainable:
            self.model.eval()
        self.hidden_size = (
            self.model.config.hidden_size
            if hasattr(self.model.config, "hidden_size")
            else self.model.config.d_model
        )

    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        context = nullcontext() if self.trainable else torch.no_grad()
        with context:
            outputs = self.model(**inputs, output_hidden_states=False, use_cache=False)
            hidden = outputs.last_hidden_state
        return hidden.detach() if not self.trainable else hidden, inputs["attention_mask"]


class AudioDiffusionGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, cfg.decoder_d_model, bias=False)
        self.duration_head = nn.Linear(self.text_encoder.hidden_size, 1)
        self.flow = MelTransformerDecoder(cfg)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_memory: torch.Tensor,
        text_attention_mask: torch.Tensor | None,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.flow(x_t, t, text_memory, text_attention_mask, mel_mask).float()

    def losses(
        self, y_logmel, texts: list[str] | None = None, mel_mask: torch.Tensor | None = None
    ):
        if texts is None:
            texts = [""] * y_logmel.shape[0]
        if mel_mask is None:
            mel_mask = torch.ones(y_logmel.shape[:2], device=y_logmel.device, dtype=torch.bool)
        hidden, text_attention_mask = self.text_encoder(texts, y_logmel.device)
        text_weights = text_attention_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(
            -1
        )
        pooled = (hidden.float() * text_weights).sum(dim=1) / text_weights.sum(dim=1).clamp_min(1.0)
        log_duration_pred = (
            self.duration_head(pooled.to(dtype=self.duration_head.weight.dtype)).squeeze(-1).float()
        )
        duration_sec = mel_mask.float().sum(dim=1).clamp_min(1.0) * (
            self.cfg.hop_length / self.cfg.sample_rate
        )
        duration_loss = F.mse_loss(log_duration_pred, duration_sec.log())
        flow_hidden = hidden
        flow_text_attention_mask = text_attention_mask
        if self.training and self.cfg.text_condition_dropout > 0:
            drop = torch.rand(len(texts), device=y_logmel.device) < self.cfg.text_condition_dropout
            if drop.any():
                flow_texts = ["" if drop[i].item() else text for i, text in enumerate(texts)]
                flow_hidden, flow_text_attention_mask = self.text_encoder(
                    flow_texts, y_logmel.device
                )
        flow_text_memory = self.text_proj(flow_hidden.to(dtype=self.text_proj.weight.dtype))
        y_norm = normalize_logmel(y_logmel, self.cfg.mel_mean, self.cfg.mel_std)
        if self.cfg.diffusion_train_steps < 2:
            raise ValueError("diffusion_train_steps must be >= 2")
        clean_weight = torch.sigmoid(
            torch.randn(y_norm.shape[0], 1, 1, device=y_norm.device) * self.cfg.timestep_logit_std
            + self.cfg.timestep_logit_mean
        )
        noise = torch.randn_like(y_norm)
        valid = mel_mask.float().unsqueeze(-1)
        timestep = ((1.0 - clean_weight) * (self.cfg.diffusion_train_steps - 1)).round()
        timestep = timestep.clamp(0, self.cfg.diffusion_train_steps - 1)
        t = timestep / (self.cfg.diffusion_train_steps - 1)
        x_t = (1 - t) * y_norm + t * noise
        pred = self.predict_velocity(x_t, t, flow_text_memory, flow_text_attention_mask, mel_mask)
        target = (noise - y_norm).float()
        flow_loss = ((pred - target).pow(2) * valid).sum() / (
            valid.sum().clamp_min(1.0) * y_norm.shape[-1]
        )
        loss = flow_loss + self.cfg.lambda_duration * duration_loss
        return (
            loss,
            flow_loss.detach(),
            duration_loss.detach(),
            log_duration_pred.exp().mean().detach(),
            duration_sec.mean().detach(),
        )

    def parameter_summary(self) -> dict[str, int]:
        text_embedding = self.text_encoder.model.get_input_embeddings()
        text_total = sum(param.numel() for param in self.text_encoder.parameters())
        text_embedding_total = sum(param.numel() for param in text_embedding.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        total = sum(param.numel() for param in self.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "text_encoder_total": text_total,
            "text_embedding_total": text_embedding_total,
            "text_encoder_body": max(0, text_total - text_embedding_total),
            "text_projector": sum(param.numel() for param in self.text_proj.parameters()),
            "duration_head": sum(param.numel() for param in self.duration_head.parameters()),
            "mel_decoder": sum(param.numel() for param in self.flow.parameters()),
        }
