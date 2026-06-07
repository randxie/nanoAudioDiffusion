from __future__ import annotations

import torch
import torch.nn.functional as F

BIGVGAN_SAMPLE_RATE = 22050
BIGVGAN_N_FFT = 1024
BIGVGAN_HOP_LENGTH = 256
BIGVGAN_WIN_LENGTH = 1024
BIGVGAN_N_MELS = 80
BIGVGAN_F_MIN = 0.0
BIGVGAN_F_MAX = 8000.0
LJSPEECH_BIGVGAN_LOGMEL_MEAN = -5.517476949990191
LJSPEECH_BIGVGAN_LOGMEL_STD = 2.06437456804736
BIGVGAN_LOGMEL_MEAN = LJSPEECH_BIGVGAN_LOGMEL_MEAN
BIGVGAN_LOGMEL_STD = LJSPEECH_BIGVGAN_LOGMEL_STD

_MEL_BASIS = {}
_HANN_WINDOW = {}


def logmel_from_wav(
    wav,
    sample_rate=BIGVGAN_SAMPLE_RATE,
    n_fft=BIGVGAN_N_FFT,
    hop_length=BIGVGAN_HOP_LENGTH,
    win_length=BIGVGAN_WIN_LENGTH,
    n_mels=BIGVGAN_N_MELS,
    f_min=BIGVGAN_F_MIN,
    f_max=BIGVGAN_F_MAX,
):
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if wav.dim() != 1:
        raise ValueError(f"expected mono waveform with shape [time], got {tuple(wav.shape)}")

    key = (sample_rate, n_fft, n_mels, float(f_min), float(f_max), wav.device)
    if key not in _MEL_BASIS:
        from librosa.filters import mel as librosa_mel_fn

        basis = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)
        _MEL_BASIS[key] = torch.from_numpy(basis).float().to(wav.device)
    if (win_length, wav.device) not in _HANN_WINDOW:
        _HANN_WINDOW[(win_length, wav.device)] = torch.hann_window(win_length, device=wav.device)

    wav = wav.float()
    pad = int((n_fft - hop_length) / 2)
    wav = F.pad(wav[None, None], (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        wav,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=_HANN_WINDOW[(win_length, wav.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    mel = torch.matmul(_MEL_BASIS[key], mag)
    return torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2).squeeze(0)


def normalize_logmel(logmel, mean=BIGVGAN_LOGMEL_MEAN, std=BIGVGAN_LOGMEL_STD):
    return (logmel.float() - mean) / std


def denormalize_logmel(logmel_norm, mean=BIGVGAN_LOGMEL_MEAN, std=BIGVGAN_LOGMEL_STD):
    return logmel_norm.float() * std + mean
