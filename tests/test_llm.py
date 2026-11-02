import pytest

from voicetutor import config, llm


def test_breaker_opens_after_failures_and_recovers():
    now = [0.0]
    b = llm.Breaker(3, 60, clock=lambda: now[0])
    for _ in range(3):
        assert b.allow()
        b.failure()
    assert not b.allow()
    now[0] = 61
    assert b.allow()


def test_success_resets_failure_count():
    b = llm.Breaker(2, 60, clock=lambda: 0)
    b.failure()
    b.success()
    b.failure()
    assert b.allow()


def test_call_groq_without_key_raises_unavailable():
    with pytest.raises(llm.GroqUnavailable, match="no key"):
        llm.call_groq(lambda: "x")


def test_call_groq_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    attempts = iter([RuntimeError, RuntimeError, None])

    def flaky():
        e = next(attempts)
        if e:
            raise e("429")
        return "ok"
    assert llm.call_groq(flaky) == "ok"


def test_three_failed_calls_open_breaker(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    for _ in range(3):
        with pytest.raises(llm.GroqUnavailable):
            llm.call_groq(lambda: 1 / 0)
    with pytest.raises(llm.GroqUnavailable, match="cooling"):
        llm.call_groq(lambda: "ok")
