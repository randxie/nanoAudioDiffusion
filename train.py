import argparse
import math
import os
import signal
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from config import load_config
from data import audio_collate, make_dataset
from evaluator import LightweightEvaluator
from metric import MetricLogger
from model import AudioDiffusionGPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--effective_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--cosine_lr", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--lr_schedule_steps", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--lambda_duration", type=float, default=None)
    p.add_argument("--text_condition_dropout", type=float, default=None)
    p.add_argument("--mel_mean", type=float, default=None)
    p.add_argument("--mel_std", type=float, default=None)
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=None)
    p.add_argument("--eval_interval", "--eval-interval", type=int, default=None)
    p.add_argument("--eval_prompts", "--eval-prompts", default=None)
    p.add_argument("--eval_num_samples", "--eval-num-samples", type=int, default=None)
    p.add_argument("--eval_steps", "--eval-steps", type=int, default=None)
    p.add_argument("--eval_cfg_scale", "--eval-cfg-scale", type=float, default=None)
    p.add_argument("--eval_whisper_model", "--eval-whisper-model", default=None)
    p.add_argument("--eval_whisper_device", "--eval-whisper-device", default=None)
    p.add_argument("--eval_vocoder_model", "--eval-vocoder-model", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--max_train_seconds", type=float, default=None)
    p.add_argument("--text_encoder_name", default=None)
    p.add_argument("--max_text_length", type=int, default=None)
    p.add_argument("--overfit_first_n_samples", type=int, default=None)
    p.add_argument("--silence_prob", type=float, default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb_url", "--wandb-url", default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cfg = load_config(args.config)
    for name in (
        "max_steps",
        "data_dir",
        "out_dir",
        "batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "lr",
        "min_lr",
        "warmup_steps",
        "lr_schedule_steps",
        "max_epochs",
        "lambda_duration",
        "text_condition_dropout",
        "mel_mean",
        "mel_std",
        "log_interval",
        "save_interval",
        "eval_interval",
        "eval_prompts",
        "eval_num_samples",
        "eval_steps",
        "eval_cfg_scale",
        "eval_whisper_model",
        "eval_whisper_device",
        "eval_vocoder_model",
        "num_workers",
        "prefetch_factor",
        "max_train_seconds",
        "text_encoder_name",
        "max_text_length",
        "overfit_first_n_samples",
        "silence_prob",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.cosine_lr:
        cfg.cosine_lr = True

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if cfg.effective_batch_size:
        per_accum_batch = cfg.batch_size * world_size
        if cfg.effective_batch_size % per_accum_batch != 0:
            raise ValueError(
                f"effective_batch_size={cfg.effective_batch_size} must be divisible by "
                f"batch_size * world_size = {per_accum_batch}"
            )
        cfg.grad_accum_steps = cfg.effective_batch_size // per_accum_batch
    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    effective_batch_size = cfg.batch_size * cfg.grad_accum_steps * world_size
    steps_per_epoch = 0
    if cfg.max_epochs > 0 and args.max_steps is None:
        if cfg.overfit_first_n_samples <= 0:
            raise ValueError("max_epochs requires overfit_first_n_samples > 0")
        steps_per_epoch = math.ceil(cfg.overfit_first_n_samples / effective_batch_size)
        cfg.max_steps = cfg.max_epochs * steps_per_epoch

    cuda_ok = torch.cuda.is_available() and (not ddp or torch.cuda.device_count() >= world_size)
    if ddp:
        dist.init_process_group(backend=os.environ.get("TORCH_DDP_BACKEND", "gloo"))
    device = torch.device(f"cuda:{local_rank}" if cuda_ok else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg.seed + rank)
    model = AudioDiffusionGPT(cfg).to(device)
    resume_state = None
    start_step = 0
    resume_elapsed_sec = 0.0
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        saved_lr_schedule_steps = int(resume_state.get("config", {}).get("lr_schedule_steps", 0))
        if args.lr_schedule_steps is None and saved_lr_schedule_steps > 0:
            cfg.lr_schedule_steps = saved_lr_schedule_steps
        model.load_state_dict(resume_state["model"])
        start_step = int(resume_state.get("step", 0))
        resume_elapsed_sec = float(resume_state.get("elapsed_sec", 0.0))
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        cuda_states = resume_state.get("cuda_rng_state_all")
        if device.type == "cuda" and cuda_states:
            cuda_state = (
                cuda_states[min(local_rank, len(cuda_states) - 1)].detach().cpu().to(torch.uint8)
            )
            torch.cuda.set_rng_state(cuda_state, device=device)
    if cfg.compile:
        model = torch.compile(model)
    train_model = (
        DDP(model, device_ids=[local_rank])
        if ddp and device.type == "cuda"
        else DDP(model)
        if ddp
        else model
    )
    if cfg.lr_schedule_steps <= 0:
        cfg.lr_schedule_steps = cfg.max_steps
    muon_params = []
    adamw_params = []
    muon_param_count = adamw_param_count = 0
    for _, param in train_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
            muon_param_count += param.numel()
        else:
            adamw_params.append(param)
            adamw_param_count += param.numel()
    optimizers = []
    optimizer_names = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                muon_params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        )
        optimizer_names.append("muon")
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=cfg.lr, weight_decay=0.0))
        optimizer_names.append("adamw")
    if resume_state is not None and "optimizers" in resume_state:
        for name, opt in zip(optimizer_names, optimizers, strict=True):
            opt.load_state_dict(resume_state["optimizers"][name])
    if rank == 0:
        print(
            f"optimizer muon_params={muon_param_count:,} adamw_params={adamw_param_count:,} "
            f"lr_schedule_steps={cfg.lr_schedule_steps}",
            flush=True,
        )
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": audio_collate,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(make_dataset(cfg, rank, world_size), **loader_kwargs)
    it = iter(loader)
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and cfg.dtype == "bf16",
    )
    logger = None
    evaluator = None
    if rank == 0:
        default_run_name = os.path.splitext(os.path.basename(args.config or "default"))[0]
        logger = MetricLogger(
            cfg.out_dir,
            args.run_name or default_run_name,
            vars(cfg),
            use_wandb=not args.no_wandb,
            wandb_url=args.wandb_url,
        )
        if cfg.eval_interval > 0:
            evaluator = LightweightEvaluator(cfg, device)
            print(
                f"eval interval={cfg.eval_interval} prompts={len(evaluator.prompts)} "
                f"steps={cfg.eval_steps} whisper={cfg.eval_whisper_model or 'off'}",
                flush=True,
            )

    train_start = time.perf_counter() - resume_elapsed_sec
    try:
        for step in range(start_step, cfg.max_steps):
            if step < cfg.warmup_steps:
                lr = cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
            elif cfg.cosine_lr:
                progress = (step + 1 - cfg.warmup_steps) / max(
                    1, cfg.lr_schedule_steps - cfg.warmup_steps
                )
                progress = min(1.0, max(0.0, progress))
                lr = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))
            else:
                lr = cfg.lr
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr
            should_log = rank == 0 and (step % cfg.log_interval == 0 or step == cfg.max_steps - 1)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            total = flow_total = duration_total = pred_duration_total = target_duration_total = 0.0
            valid_frames_total = 0
            for micro_step in range(cfg.grad_accum_steps):
                sync_context = (
                    train_model.no_sync()
                    if ddp and micro_step < cfg.grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    mel, mel_mask, texts = next(it)
                    mel = mel.to(device, non_blocking=True)
                    mel_mask = mel_mask.to(device, non_blocking=True)
                    with autocast:
                        if ddp:
                            loss, flow_loss, duration, pred_duration, target_duration = (
                                train_model.module.losses(mel, texts, mel_mask)
                            )
                        else:
                            loss, flow_loss, duration, pred_duration, target_duration = (
                                train_model.losses(mel, texts, mel_mask)
                            )
                        loss = loss / cfg.grad_accum_steps
                    loss.backward()
                total += loss.detach().float().item()
                flow_total += flow_loss.float().item() / cfg.grad_accum_steps
                duration_total += duration.float().item() / cfg.grad_accum_steps
                pred_duration_total += pred_duration.float().item() / cfg.grad_accum_steps
                target_duration_total += target_duration.float().item() / cfg.grad_accum_steps
                valid_frames_total += int(mel_mask.sum().item())
            grad_norm = None
            if should_log:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    train_model.parameters(), float("inf")
                ).item()
            for opt in optimizers:
                opt.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_sec = time.perf_counter() - step_start

            if should_log:
                frames_per_step = valid_frames_total * world_size
                metrics = {
                    "train/loss": total,
                    "train/flow_loss": flow_total,
                    "train/duration_loss": duration_total,
                    "train/duration_pred_sec": pred_duration_total,
                    "train/duration_target_sec": target_duration_total,
                    "train/micro_batch_size": cfg.batch_size,
                    "train/grad_accum_steps": cfg.grad_accum_steps,
                    "train/effective_batch_size": effective_batch_size,
                    "train/valid_frames": frames_per_step,
                    "time/step_sec": step_sec,
                    "perf/samples_per_sec": effective_batch_size / step_sec,
                    "perf/frames_per_sec": frames_per_step / step_sec,
                    "perf/mel_values_per_sec": frames_per_step * cfg.n_mels / step_sec,
                    "optim/lr": optimizers[0].param_groups[0]["lr"],
                    "optim/grad_norm": grad_norm,
                }
                if steps_per_epoch:
                    metrics["train/epoch"] = (step + 1) / steps_per_epoch
                if device.type == "cuda":
                    metrics["cuda/max_memory_gb"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                logger.log(metrics, step)
                logger.print_train(metrics, step)

            completed_step = step + 1
            if rank == 0 and not args.no_save and completed_step % cfg.save_interval == 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                state = {
                    "model": model.state_dict(),
                    "optimizers": {
                        name: opt.state_dict()
                        for name, opt in zip(optimizer_names, optimizers, strict=True)
                    },
                    "config": vars(cfg),
                    "step": completed_step,
                    "elapsed_sec": time.perf_counter() - train_start,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                }
                latest_tmp = os.path.join(cfg.out_dir, "latest.pt.tmp")
                torch.save(state, latest_tmp)
                os.replace(latest_tmp, os.path.join(cfg.out_dir, "latest.pt"))
                step_path = os.path.join(cfg.out_dir, f"ckpt_step_{completed_step:08d}.pt")
                step_tmp = step_path + ".tmp"
                torch.save(state, step_tmp)
                os.replace(step_tmp, step_path)

            should_eval = cfg.eval_interval > 0 and completed_step % cfg.eval_interval == 0
            if should_eval and ddp:
                dist.barrier()
            if should_eval and rank == 0:
                eval_model = train_model.module if ddp else train_model
                eval_metrics = evaluator.run(eval_model, completed_step)
                logger.log(eval_metrics, completed_step)
                parts = [f"eval step {completed_step}"]
                if "eval/wer" in eval_metrics:
                    parts.append(f"wer {eval_metrics['eval/wer']:.4f}")
                parts.append(f"path {eval_metrics['eval/path']}")
                print(" ".join(parts), flush=True)
            if should_eval and ddp:
                dist.barrier()

            should_stop = (
                stop_requested
                or os.path.exists(os.path.join(cfg.out_dir, "STOP"))
                or (
                    cfg.max_train_seconds > 0
                    and time.perf_counter() - train_start >= cfg.max_train_seconds
                )
            )
            if ddp:
                stop_flag = torch.tensor(1 if should_stop else 0, device=device)
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                should_stop = bool(stop_flag.item())
            if should_stop:
                break

        final_step = step + 1 if start_step < cfg.max_steps else start_step
        if rank == 0 and not args.no_save:
            os.makedirs(cfg.out_dir, exist_ok=True)
            state = {
                "model": model.state_dict(),
                "optimizers": {
                    name: opt.state_dict()
                    for name, opt in zip(optimizer_names, optimizers, strict=True)
                },
                "config": vars(cfg),
                "step": final_step,
                "elapsed_sec": time.perf_counter() - train_start,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            }
            latest_tmp = os.path.join(cfg.out_dir, "latest.pt.tmp")
            torch.save(state, latest_tmp)
            os.replace(latest_tmp, os.path.join(cfg.out_dir, "latest.pt"))
            torch.save(state, os.path.join(cfg.out_dir, "ckpt.pt"))
    finally:
        if logger is not None:
            logger.close()
        if ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
