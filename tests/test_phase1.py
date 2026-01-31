from __future__ import annotations

import os
import wave

import pytest

from rayado import phase1
from rayado.models import Segment


def _write_wav(path: str, seconds: int = 1) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * seconds)


def test_group_segments_merges_target_max():
    segments = [
        Segment(0.0, 4.0),
        Segment(4.1, 8.0),
        Segment(8.2, 12.0),
    ]
    grouped = phase1._group_segments(segments, target_sec=10.0, max_sec=12.0)
    assert len(grouped) == 1
    assert grouped[0].start == 0.0
    assert grouped[0].end == 12.0


def test_phase1_runs_with_mocks(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    out_srt = tmp_path / "out.srt"

    def fake_extract_audio_file(inp, out, sample_rate=16000, channels=1):
        _write_wav(out, seconds=2)

    def fake_extract_segment(inp, out, start, end, sample_rate=16000, channels=1):
        _write_wav(out, seconds=1)

    def fake_ffprobe_duration(_):
        return 10.0

    def fake_silencedetect(*args, **kwargs):
        return []

    def fake_detect_language(wavs, cache_dir, device="cpu"):
        return "zh", {"zh": 1.0}, []

    def fake_transcribe_chunk(**kwargs):
        return ([], {"words": []})

    monkeypatch.setattr(phase1, "extract_audio_file", fake_extract_audio_file)
    monkeypatch.setattr(phase1, "extract_audio_file_segment", fake_extract_segment)
    monkeypatch.setattr(phase1, "ffprobe_duration", fake_ffprobe_duration)
    monkeypatch.setattr(phase1, "silencedetect", fake_silencedetect)
    monkeypatch.setattr(phase1, "detect_language_voxlingua", fake_detect_language)
    monkeypatch.setattr(phase1, "transcribe_chunk", fake_transcribe_chunk)

    language = phase1.run_phase1(
        input_path=str(input_path),
        out_srt_path=str(out_srt),
        cache_dir=str(tmp_path / ".cache"),
        concurrency=2,
        retry=1,
        deepgram_model="nova-2",
        vad_threshold=-30.0,
        vad_min_speech_sec=0.6,
        vad_merge_gap_sec=0.3,
        vad_pad_sec=0.1,
        vad_min_silence_sec=0.5,
        target_sec=2.0,
        max_sec=3.0,
        lid_cache_dir=str(tmp_path / ".lid"),
        lid_device="cpu",
    )

    assert language == "zh"
    assert out_srt.exists()
