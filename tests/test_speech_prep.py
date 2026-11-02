import pytest

from voicetutor import tts
from voicetutor.tts import rate_string, speech_prep


@pytest.mark.parametrize("raw,expected", [
    ("**Bold** and _italic_ text", "Bold and italic text"),
    ("## Heading\n- first point\n- second point", "Heading. first point. second point."),
    ("1. one\n2. two", "one. two."),
    ("| a | b |\n|---|---|\n| 1 | 2 |", "a, b. 1, 2."),
    ("WER = (S + D + I) / N", "WER equals S plus D plus I over N"),
    ("Accuracy is 95%", "Accuracy is 95 percent"),
    ("x^2 + y^3", "x squared plus y cubed"),
    ("See pages 3-5, e.g. MFCCs", "See pages 3 to 5, for example MFCCs"),
    ("a <= b and c != d", "a less than or equal to b and c not equal to d"),
    ("[link](http://x.com) and `code`", "link and code"),
])
def test_speech_prep_cases(raw, expected):
    assert speech_prep(raw).rstrip(".") == expected.rstrip(".")


def test_speech_prep_splits_long_sentences_and_semicolons():
    long = ", ".join(f"item {i}" for i in range(20)) + " end."
    out = speech_prep(long + " A; B")
    assert all(len(s.split()) <= 25 for s in out.split(". "))
    assert "A. B" in out


def test_speech_prep_empty():
    assert speech_prep("") == ""


def test_rate_string():
    assert rate_string(-20) == "-20%" and rate_string(15) == "+15%"


def test_synthesize_falls_back_in_order(monkeypatch):
    calls = []

    def fail(name):
        def f(text, rate):
            calls.append(name)
            raise RuntimeError
        return f
    monkeypatch.setattr(tts, "_edge", fail("edge"))
    monkeypatch.setattr(tts, "_gtts", lambda t, r: calls.append("gtts") or "x.mp3")
    assert tts.synthesize("hello") == ("x.mp3", "gtts")
    assert calls == ["edge", "gtts"]


def test_synthesize_never_raises(monkeypatch):
    for n in ("_edge", "_gtts", "_pyttsx3"):
        monkeypatch.setattr(tts, n, lambda t, r: 1 / 0)
    assert tts.synthesize("hello") == (None, "none")
    assert tts.synthesize("  ") == (None, "none")
