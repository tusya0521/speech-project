import numpy as np
import pytest

from voicetutor import asr, config, llm


def tone(seconds=1.0):
    t = np.arange(int(seconds * asr.SAMPLE_RATE)) / asr.SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def wav(tmp_path, monkeypatch):
    path = str(tmp_path / "q.wav")
    asr.write_wav(tone(), path)
    monkeypatch.setattr(asr, "load_audio", lambda p: tone())
    return path


SEG = [{"text": " What is MFCC?", "start": 0.0, "end": 1.0, "avg_logprob": -0.2, "no_speech_prob": 0.01}]


@pytest.mark.parametrize("text,lp,ns,expected", [
    ("what is mfcc", -0.2, 0.01, "accept"),
    ("what is mfcc", -0.7, 0.01, "confirm"),
    ("what is mfcc", -1.5, 0.01, "reject"),
    ("what is mfcc", -0.2, 0.9, "reject"),
    ("", -0.1, 0.0, "reject"),
])
def test_decide_thresholds(text, lp, ns, expected):
    assert asr.decide(text, lp, ns) == expected


def test_aggregate_weights_by_duration():
    segs = [{"text": "a", "start": 0, "end": 3, "avg_logprob": -0.1, "no_speech_prob": 0.0},
            {"text": "b", "start": 3, "end": 4, "avg_logprob": -0.9, "no_speech_prob": 0.4}]
    text, lp, ns = asr.aggregate(segs)
    assert text == "a b" and abs(lp - -0.3) < 1e-9 and abs(ns - 0.1) < 1e-9
    assert asr.aggregate([]) == ("", -10.0, 1.0)


def test_vocabulary_prompt_is_capped():
    p = asr.vocabulary_prompt(["hidden markov"] * 200)
    assert p.startswith("Study notes vocabulary:") and len(p.split()) < 130
    assert asr.vocabulary_prompt([]) == ""


def test_silence_rejected_without_calling_any_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(asr, "_local", lambda *a: pytest.fail("should not run"))
    r = asr.transcribe("x.wav")
    assert r.decision == "reject" and "silence" in r.note


def test_unreadable_audio_is_handled(monkeypatch):
    monkeypatch.setattr(asr, "load_audio", lambda p: 1 / 0)
    assert asr.transcribe("broken.wav").decision == "reject"
    assert asr.transcribe("").decision == "reject"


def test_no_key_goes_straight_to_local(wav, monkeypatch):
    monkeypatch.setattr(asr, "_groq", lambda *a: pytest.fail("no key: groq must not be called"))
    monkeypatch.setattr(asr, "_local", lambda a, p: SEG)
    r = asr.transcribe(wav, ["mfcc"])
    assert (r.backend, r.decision, r.text, r.note) == ("local", "accept", "What is MFCC?", "")


def test_groq_failure_retries_then_falls_back(wav, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test")
    calls = {"n": 0}

    class FakeClient:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kw):
                    calls["n"] += 1
                    raise ConnectionError("down")
    monkeypatch.setattr(llm, "client", lambda: FakeClient)
    monkeypatch.setattr(asr, "_local", lambda a, p: SEG)
    r = asr.transcribe(wav)
    assert calls["n"] == config.GROQ_RETRIES + 1
    assert r.backend == "local" and "offline" in r.note


def test_groq_success_passes_prompt(wav, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test")
    seen = {}

    class Resp:
        segments = SEG

    class FakeClient:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return Resp
    monkeypatch.setattr(llm, "client", lambda: FakeClient)
    r = asr.transcribe(wav, ["viterbi"])
    assert r.backend == "groq" and "viterbi" in seen["prompt"] and seen["response_format"] == "verbose_json"


def test_overlong_audio_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "load_audio", lambda p: tone(config.MAX_AUDIO_SECONDS + 5))
    got = {}
    monkeypatch.setattr(asr, "_local", lambda a, p: got.setdefault("n", a.size) and SEG)
    r = asr.transcribe(str(tmp_path / "long.wav"))
    assert got["n"] == int(config.MAX_AUDIO_SECONDS * asr.SAMPLE_RATE)
    assert "longer than" in r.note
