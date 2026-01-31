from __future__ import annotations

import sys
import types

import torch

from rayado import lid_voxlingua


def test_pick_samples_vote(monkeypatch):
    class FakeClassifier:
        def classify_batch(self, _signal):
            return (None, torch.tensor([0.0]), None, ["zh: Chinese"])

    def fake_from_hparams(*args, **kwargs):
        return FakeClassifier()

    def fake_load_wav(_path):
        return torch.zeros((1, 16000))

    fake_fetching = types.SimpleNamespace(LocalStrategy=types.SimpleNamespace(COPY="copy"))
    fake_inference = types.SimpleNamespace(
        classifiers=types.SimpleNamespace(EncoderClassifier=types.SimpleNamespace(from_hparams=fake_from_hparams))
    )
    fake_speechbrain = types.SimpleNamespace(utils=types.SimpleNamespace(fetching=fake_fetching), inference=fake_inference)

    sys.modules["speechbrain"] = fake_speechbrain
    sys.modules["speechbrain.utils"] = fake_speechbrain.utils
    sys.modules["speechbrain.utils.fetching"] = fake_fetching
    sys.modules["speechbrain.inference"] = fake_inference
    sys.modules["speechbrain.inference.classifiers"] = fake_inference.classifiers

    monkeypatch.setattr(lid_voxlingua, "_load_wav_tensor", fake_load_wav)
    monkeypatch.setattr(lid_voxlingua, "_patch_torchaudio_backend", lambda: None)
    monkeypatch.setattr(lid_voxlingua, "_patch_huggingface_hub", lambda: None)
    monkeypatch.setattr(lid_voxlingua, "_patch_speechbrain_fetch", lambda: None)

    wavs = [f"seg_{i}.wav" for i in range(10)]
    winner, weights, samples = lid_voxlingua.detect_language_voxlingua(
        wavs, cache_dir=".cache", device="cpu"
    )
    assert winner == "zh"
    assert "zh" in weights
    assert len(samples) == 5
