"""Speech recognition: Groq Whisper first, local faster-whisper as fallback, plus confidence scoring."""
import os
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config, llm

SAMPLE_RATE = 16000


@dataclass
class ASRResult:
    text: str
    avg_logprob: float
    no_speech_prob: float
    backend: str            # "groq", "local" or "none"
    decision: str           # "accept", "confirm" or "reject"
    note: str = ""          # user-facing message (e.g. audio was trimmed, fell back to offline)


def vocabulary_prompt(keywords: list[str], max_words: int = 120) -> str:
    """Whisper 'prompt' that biases decoding toward the document's technical terms.

    Groq caps the prompt at 224 tokens, so we keep it to about 120 words.
    """
    out, n = [], 0
    for k in keywords:
        n += len(k.split())
        if n > max_words:
            break
        out.append(k)
    return ("Study notes vocabulary: " + ", ".join(out) + ".") if out else ""


def decide(text: str, avg_logprob: float, no_speech_prob: float) -> str:
    """Three-way confidence decision from Whisper's own scores and transcript length."""
    if len(text.split()) < config.ASR_MIN_WORDS or no_speech_prob > config.ASR_NO_SPEECH_REJECT:
        return "reject"
    if avg_logprob < config.ASR_REJECT_LOGPROB:
        return "reject"
    if avg_logprob < config.ASR_ACCEPT_LOGPROB:
        return "confirm"
    return "accept"


def aggregate(segments: list[dict]) -> tuple[str, float, float]:
    """Join segment texts; average avg_logprob and no_speech_prob weighted by segment duration."""
    if not segments:
        return "", -10.0, 1.0
    w = np.array([max(s["end"] - s["start"], 0.01) for s in segments])
    lp = float(np.average([s["avg_logprob"] for s in segments], weights=w))
    ns = float(np.average([s["no_speech_prob"] for s in segments], weights=w))
    return " ".join(s["text"].strip() for s in segments).strip(), lp, ns


def load_audio(path: str) -> np.ndarray:
    from faster_whisper import decode_audio  # PyAV based: no system ffmpeg needed
    return decode_audio(path, sampling_rate=SAMPLE_RATE)


def write_wav(audio: np.ndarray, path: str):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def _groq(path: str, prompt: str) -> list[dict]:
    with open(path, "rb") as f:
        data = f.read()
    kwargs = dict(file=(os.path.basename(path), data), model=config.GROQ_ASR_MODEL, response_format="verbose_json",
                  language="en", temperature=0.0)
    if prompt:
        kwargs["prompt"] = prompt
    resp = llm.call_groq(lambda: llm.client().audio.transcriptions.create(**kwargs))
    segs = getattr(resp, "segments", None) or []
    return [s if isinstance(s, dict) else s.model_dump() for s in segs]


@lru_cache(maxsize=1)
def _local_model():
    from faster_whisper import WhisperModel
    device = config.LOCAL_WHISPER_DEVICE
    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device == "cuda":
        try:
            return WhisperModel(config.LOCAL_WHISPER_MODEL, device="cuda", compute_type="float16")
        except Exception as e:  # usually missing cuBLAS/cuDNN DLLs; CPU still works
            print(f"[asr] CUDA load failed ({e}); using CPU")
    return WhisperModel(config.LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8")


def local_device() -> str:
    """Which device the loaded local model actually uses (for logs and the eval environment line)."""
    return _local_model().model.device


def _local(audio: np.ndarray, prompt: str) -> list[dict]:
    segs, _ = _local_model().transcribe(audio, language="en", initial_prompt=prompt or None,
                                        beam_size=5, temperature=0.0, vad_filter=True)
    return [{"text": s.text, "start": s.start, "end": s.end, "avg_logprob": s.avg_logprob,
             "no_speech_prob": s.no_speech_prob} for s in segs]


def transcribe(path: str, keywords: list[str] | None = None, backend: str = "auto") -> ASRResult:
    """Transcribe an audio file. backend: 'auto' (Groq then local), 'groq' or 'local'. Never raises."""
    try:
        audio = load_audio(path) if path else np.zeros(0, dtype=np.float32)
    except Exception:
        return ASRResult("", -10, 1, "none", "reject", "I could not read that recording.")
    if audio.size == 0 or float(np.sqrt(np.mean(audio ** 2))) < 1e-3:
        return ASRResult("", -10, 1, "none", "reject", "I heard only silence.")

    note, clip = "", path
    max_samples = int(config.MAX_AUDIO_SECONDS * SAMPLE_RATE)
    if audio.size > max_samples:
        audio = audio[:max_samples]
        note = f"Your recording was longer than {config.MAX_AUDIO_SECONDS:.0f} seconds, so I only used the start."
    tmp = None
    if note or not path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        write_wav(audio, tmp)
        clip = tmp

    prompt = vocabulary_prompt(keywords or [])
    segments, used = None, "none"
    try:
        if backend == "groq" or (backend == "auto" and config.GROQ_API_KEY):
            try:
                segments, used = _groq(clip, prompt), "groq"
            except llm.GroqUnavailable as e:
                if backend == "groq":
                    raise
                if str(e) != "no key":
                    note = (note + " Online speech recognition failed, so I used the offline model.").strip()
        if segments is None:
            segments, used = _local(audio, prompt), "local"
    except Exception:
        return ASRResult("", -10, 1, used, "reject", "Speech recognition failed. Please type your question.")
    finally:
        if tmp:
            os.remove(tmp)

    text, lp, ns = aggregate(segments)
    return ASRResult(text, lp, ns, used, decide(text, lp, ns), note)
