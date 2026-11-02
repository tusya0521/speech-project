import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from voicetutor import config, llm, retrieval  # noqa: E402


def fake_embed(texts, query=False):
    """Deterministic bag-of-words hashing embedding so tests don't download a model."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in retrieval.tokenize(t):
            out[i, hash(w) % 256] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """No network, no real model, fresh breaker, metrics into tmp, no sleeping."""
    monkeypatch.setattr(retrieval, "embed", fake_embed)
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_BACKOFF_S", 0.0)
    monkeypatch.setattr(config, "METRICS_PATH", str(tmp_path / "lat.jsonl"))
    monkeypatch.setattr(config, "RETRIEVAL_MIN_CONFIDENCE", 0.2)
    monkeypatch.setattr(llm, "breaker", llm.Breaker(3, 60))
    monkeypatch.setattr(llm, "_client", None)
