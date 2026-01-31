from __future__ import annotations

import os
import re
import wave
from array import array
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


def _patch_torchaudio_backend() -> None:
    try:
        import torchaudio
    except Exception:
        return
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend: None  # type: ignore[attr-defined]


def _patch_huggingface_hub() -> None:
    try:
        import huggingface_hub
    except Exception:
        return
    if not hasattr(huggingface_hub, "hf_hub_download"):
        return
    if "use_auth_token" not in huggingface_hub.hf_hub_download.__code__.co_varnames:
        original_hf_download = huggingface_hub.hf_hub_download

        def _hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return original_hf_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _hf_hub_download_compat  # type: ignore[assignment]


def _patch_speechbrain_fetch() -> None:
    try:
        import speechbrain.utils.fetching as sb_fetching
    except Exception:
        return
    if hasattr(sb_fetching, "fetch"):
        original_fetch = sb_fetching.fetch

        def _fetch_guard(filename, *args, **kwargs):  # type: ignore[no-untyped-def]
            if not filename:
                return ""
            try:
                return original_fetch(filename, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if filename == "custom.py" and ("404" in msg or "Not Found" in msg):
                    raise ValueError("optional custom.py missing") from exc
                raise

        sb_fetching.fetch = _fetch_guard  # type: ignore[assignment]


def _load_wav_tensor(path: str) -> torch.Tensor:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sampwidth == 2:
        data = array("h")
        data.frombytes(frames)
        scale = 32768.0
    elif sampwidth == 4:
        data = array("i")
        data.frombytes(frames)
        scale = 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sampwidth}")

    samples = torch.tensor(data, dtype=torch.float32) / scale
    if n_channels > 1:
        samples = samples.view(-1, n_channels).mean(dim=1)
    return samples.unsqueeze(0)


@dataclass(frozen=True)
class LidSample:
    path: str
    language: str
    score: float
    label: str


def _label_to_code(label: str) -> str:
    if not label:
        return ""
    if ":" in label:
        return label.split(":", 1)[0].strip()
    return label.split()[0].strip()


def _pick_samples(paths: List[str]) -> List[str]:
    if not paths:
        return []
    indexed: List[Tuple[float, str]] = []
    for path in paths:
        name = os.path.basename(path)
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.wav$", name)
        if match:
            start = float(match.group(1))
        else:
            start = float("inf")
        indexed.append((start, path))
    indexed.sort(key=lambda x: (x[0], x[1]))
    ordered = [path for _, path in indexed]
    n = len(ordered)
    positions = [0, int(n * 0.25), int(n * 0.50), int(n * 0.75), n - 1]
    chosen: List[str] = []
    seen = set()
    for pos in positions:
        pos = max(0, min(n - 1, pos))
        path = ordered[pos]
        if path in seen:
            continue
        seen.add(path)
        chosen.append(path)
    return chosen


def detect_language_voxlingua(
    wav_paths: List[str],
    *,
    cache_dir: str,
    device: str = "cpu",
) -> Tuple[str, Dict[str, float], List[LidSample]]:
    _patch_torchaudio_backend()
    _patch_huggingface_hub()
    _patch_speechbrain_fetch()

    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils import fetching as sb_fetching

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=cache_dir,
        run_opts={"device": device},
        local_strategy=sb_fetching.LocalStrategy.COPY,
    )

    samples = _pick_samples(wav_paths)
    votes: Dict[str, int] = {}
    weights: Dict[str, float] = {}
    details: List[LidSample] = []

    for path in samples:
        with torch.no_grad():
            signal = _load_wav_tensor(path)
            prediction = classifier.classify_batch(signal)
        score = float(prediction[1].exp().squeeze().item())
        label = prediction[3][0] if prediction[3] else ""
        code = _label_to_code(label)

        votes[code] = votes.get(code, 0) + 1
        weights[code] = weights.get(code, 0.0) + score
        details.append(LidSample(path=path, language=code, score=score, label=label))

    winner = ""
    if weights:
        winner = sorted(weights.items(), key=lambda x: (-x[1], x[0]))[0][0]
    elif votes:
        winner = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return winner, weights, details
