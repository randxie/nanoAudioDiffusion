---
name: tts-research-workflow
description: nanoAudioLLM TTS research workflow for model, objective, sampler, text encoder, mel processing, or data-path changes; validate with static checks, overfit analysis, full-run monitoring, Whisper WER, and final eval before trusting a change.
---

# TTS Research Workflow

Use this skill whenever changing nanoAudioLLM training logic, model architecture, sampler behavior, text conditioning, mel processing, optimizer settings, eval logic, or dataset code.

## 1. Change Code Surgically

- Keep the implementation minimal and direct. Do not add abstractions for one-off branches.
- Keep the release path on LJSpeech, flow matching, Euler sampling, BigVGAN-compatible log-mels, frozen text encoder, duration predictor, and Whisper WER.
- Update configs/docs only when the behavior actually changes.
- Before training, run:

```bash
.venv/bin/python -m py_compile train.py model.py config.py audio.py data.py evaluator.py metric.py sample.py scripts/*.py
.venv/bin/python scripts/model_size.py --config configs/ljspeech.yaml
.venv/bin/python scripts/model_size.py --config configs/ljspeech_overfit64.yaml
.venv/bin/python scripts/model_size.py --config configs/ljspeech.yaml --flops --batch_size 2 --text_len 128 --seq_len 128
```

If mel extraction, audio parameters, tokenizer, or dataset packing changed, recalculate stats:

```bash
.venv/bin/python scripts/dataset_stats.py --config configs/ljspeech.yaml
```

## 2. Overfit First

Run the 64-sample LJSpeech overfit before any full run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --config configs/ljspeech_overfit64.yaml
```

The overfit config must use `eval_prompts: overfit`, so WER is measured on the same 64 repeated LJSpeech prompts loaded dynamically from `data_dir`. Do not use an unrelated prompt set or a tiny canary for the overfit WER gate.

Overfit analysis must include:

- final/latest train loss, flow loss, and duration loss
- WER on all 64 overfit prompts
- generated samples from the overfit checkpoint
- whether predicted duration is plausible for those prompts
- any obvious mel/vocoder mismatch before starting a full run

Sample fixed prompts from the overfit checkpoint:

```bash
.venv/bin/python sample.py --ckpt .artifacts/local/ljspeech_byt5_overfit64/latest.pt --text "Hi, how are you?" --steps 50 --cfg_scale 2.0 --out .artifacts/local/generated_samples/overfit_hi.wav
.venv/bin/python sample.py --ckpt .artifacts/local/ljspeech_byt5_overfit64/latest.pt --text "The quick brown fox jumps over the lazy dog." --steps 50 --cfg_scale 2.0 --out .artifacts/local/generated_samples/overfit_fox.wav
.venv/bin/python sample.py --ckpt .artifacts/local/ljspeech_byt5_overfit64/latest.pt --text "Today is a good day to test speech synthesis." --steps 50 --cfg_scale 2.0 --out .artifacts/local/generated_samples/overfit_tts.wav
```

## 3. Full Run

Only start the full LJSpeech run after the overfit loss drops clearly and fixed prompt samples are intelligible enough to justify the run.

Start the standard run:

```bash
./run.sh
```

The current full config uses:

- `configs/ljspeech.yaml`
- output directory `.artifacts/local/ljspeech_byt5_flow_matching_200k`
- `google/byt5-small`
- effective batch size 64
- 200K max steps
- in-loop small-set WER eval every 1000 steps
- `eval_num_samples: 8`
- `eval_steps: 50`
- `eval_cfg_scale: 2.0`
- Whisper on CUDA

## 4. Monitor Loss And WER

During a full run, monitor:

- `train/loss`, `train/flow_loss`, `train/duration_loss`
- `eval/wer` every eval interval
- `optim/lr`
- `perf/samples_per_sec` and `time/step_sec`
- checkpoint cadence and `latest.pt`

Useful checks:

```bash
tail -n 20 .artifacts/local/ljspeech_byt5_flow_matching_200k/metrics.jsonl
.venv/bin/python scripts/plot_training_loss.py --metrics .artifacts/local/ljspeech_byt5_flow_matching_200k/metrics.jsonl --out .artifacts/local/loss_wer.png
.venv/bin/python scripts/plot_step_time.py --metrics .artifacts/local/ljspeech_byt5_flow_matching_200k/metrics.jsonl
```

If listening samples are needed mid-run, pause cleanly with the run `STOP` marker, wait for `latest.pt` to be written, sample, then resume with the same config. Do not kill the process if a clean pause is possible.

## 5. Full Eval

Run final small-set eval from the selected checkpoint. Use the same small prompt set used during monitoring unless the experiment explicitly defines another LJSpeech-local eval set.

```bash
.venv/bin/python scripts/eval_checkpoint.py \
  --ckpt .artifacts/local/ljspeech_byt5_flow_matching_200k/latest.pt \
  --prompts <small_eval_prompts.txt> \
  --out_dir .artifacts/local/eval_small_final \
  --limit 64 \
  --steps 50 \
  --cfg_scale 2.0 \
  --whisper_model base.en \
  --whisper_device cuda
```

Generate a few listening samples that are not from the eval list:

```bash
.venv/bin/python sample.py --ckpt .artifacts/local/ljspeech_byt5_flow_matching_200k/latest.pt --text "Hi, how are you?" --steps 50 --cfg_scale 2.0 --out .artifacts/local/final_samples/hi.wav
.venv/bin/python sample.py --ckpt .artifacts/local/ljspeech_byt5_flow_matching_200k/latest.pt --text "The weather is beautiful today." --steps 50 --cfg_scale 2.0 --out .artifacts/local/final_samples/weather.wav
.venv/bin/python sample.py --ckpt .artifacts/local/ljspeech_byt5_flow_matching_200k/latest.pt --text "A small model can still learn to speak clearly." --steps 50 --cfg_scale 2.0 --out .artifacts/local/final_samples/small_model.wav
```

## Reporting

Always report:

- changed files and validation commands
- checkpoint path
- latest/full-run loss and WER
- path to the loss/WER plot
- final eval `wer.jsonl` and `summary.json`
- sample wav paths
- any blocked check with the exact missing command, dependency, or artifact
