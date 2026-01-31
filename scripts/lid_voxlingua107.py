from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import torch
import wave
from array import array

# Patch torchaudio backend helpers before SpeechBrain import.
try:
    import torchaudio
except Exception:
    torchaudio = None  # type: ignore[assignment]
else:
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend: None  # type: ignore[attr-defined]

# Force SpeechBrain local strategy to "copy" at import time.
try:
    import speechbrain.utils.fetching as sb_fetching
except Exception:
    sb_fetching = None  # type: ignore[assignment]
else:
    # Leave LocalStrategy enum intact; we'll pass COPY explicitly later.
    pass

    if hasattr(sb_fetching, "fetch"):
        _orig_fetch = sb_fetching.fetch

        def _fetch_guard(filename, *args, **kwargs):  # type: ignore[no-untyped-def]
            if not filename:
                return ""
            try:
                return _orig_fetch(filename, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if filename == "custom.py" and ("404" in msg or "Not Found" in msg):
                    raise ValueError("optional custom.py missing") from exc
                raise

        sb_fetching.fetch = _fetch_guard  # type: ignore[assignment]


def _load_classifier(cache_dir: str, device: str):
    try:
        import huggingface_hub
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"huggingface_hub import failed: {exc}") from exc

    if not hasattr(huggingface_hub, "hf_hub_download"):
        raise RuntimeError("huggingface_hub.hf_hub_download is missing")

    if "use_auth_token" not in huggingface_hub.hf_hub_download.__code__.co_varnames:
        original_hf_download = huggingface_hub.hf_hub_download

        def _hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return original_hf_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _hf_hub_download_compat  # type: ignore[assignment]

    if torchaudio is None:
        raise RuntimeError("torchaudio import failed")

    try:
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils import fetching as sb_fetching
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"speechbrain import failed: {exc}") from exc

    import inspect

    base_kwargs = {
        "source": "speechbrain/lang-id-voxlingua107-ecapa",
        "savedir": cache_dir,
        "run_opts": {"device": device},
    }
    sig = inspect.signature(EncoderClassifier.from_hparams)
    if "local_strategy" in sig.parameters:
        base_kwargs["local_strategy"] = sb_fetching.LocalStrategy.COPY
    if "hparams_file" in sig.parameters:
        base_kwargs["hparams_file"] = "hyperparams.yaml"
    if "pymodule_file" in sig.parameters:
        base_kwargs["pymodule_file"] = "custom.py"

    return EncoderClassifier.from_hparams(**base_kwargs)


def _list_chunks(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".wav"):
            files.append(name)
    return files


def _extract_start_seconds(name: str) -> float | None:
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.wav$", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _pick_samples(files: List[str]) -> List[str]:
    if not files:
        return []
    indexed: List[Tuple[float, str]] = []
    for name in files:
        start = _extract_start_seconds(name)
        if start is None:
            start = float("inf")
        indexed.append((start, name))
    indexed.sort(key=lambda x: (x[0], x[1]))
    ordered = [name for _, name in indexed]
    n = len(ordered)
    positions = [0, int(n * 0.25), int(n * 0.50), int(n * 0.75), n - 1]
    chosen = []
    seen = set()
    for pos in positions:
        pos = max(0, min(n - 1, pos))
        name = ordered[pos]
        if name in seen:
            continue
        seen.add(name)
        chosen.append(name)
    return chosen


def _label_to_code(label: str) -> str:
    if not label:
        return ""
    if ":" in label:
        return label.split(":", 1)[0].strip()
    return label.split()[0].strip()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote dominant language using VoxLingua107 (SpeechBrain)")
    parser.add_argument("chunk_dir", help="Directory of WAV chunks")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(".cache", "speechbrain", "lang-id-voxlingua107-ecapa"),
        help="Model cache directory",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    args = parser.parse_args()

    # Avoid symlink creation on Windows without elevated privileges.
    os.environ.setdefault("SPEECHBRAIN_LOCAL_STRATEGY", "copy")

    chunk_dir = args.chunk_dir
    if not os.path.isdir(chunk_dir):
        print(f"Chunk dir not found: {chunk_dir}", file=sys.stderr)
        sys.exit(1)

    files = _list_chunks(chunk_dir)
    samples = _pick_samples(files)
    if not samples:
        print("No wav chunks found.", file=sys.stderr)
        sys.exit(2)

    try:
        classifier = _load_classifier(args.cache_dir, args.device)
    except RuntimeError as exc:
        print(
            "Failed to load SpeechBrain/TorchAudio. "
            "Install compatible torch/torchaudio for your Python version. "
            "Example (CPU): pip install --upgrade torch torchaudio",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        sys.exit(3)

    votes: Dict[str, int] = {}
    weights: Dict[str, float] = {}
    details = []

    for name in samples:
        path = os.path.join(chunk_dir, name)
        with torch.no_grad():
            signal = _load_wav_tensor(path)
            prediction = classifier.classify_batch(signal)
        score = float(prediction[1].exp().squeeze().item())
        label = prediction[3][0] if prediction[3] else ""
        code = _label_to_code(label)

        votes[code] = votes.get(code, 0) + 1
        weights[code] = weights.get(code, 0.0) + score
        details.append({"file": name, "language": code, "score": score, "label": label})

    winner = ""
    if weights:
        winner = sorted(weights.items(), key=lambda x: (-x[1], x[0]))[0][0]
    elif votes:
        winner = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]

    output = {
        "sample_count": len(samples),
        "samples": details,
        "votes": votes,
        "weights": weights,
        "winner": winner,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
