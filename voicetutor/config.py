"""Environment-driven settings. Every threshold lives here so it can be tuned without code edits."""
import os

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")  # routing + query rewrite

# Local ASR fallback. "auto" picks cuda if available, else cpu.
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base.en")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Upload limits (kept small for the free HF CPU tier).
MAX_UPLOAD_MB = _f("MAX_UPLOAD_MB", 20)
MAX_PAGES = _i("MAX_PAGES", 300)

# Chunking, measured in words (~1.3 tokens per word, so 300 words is ~400 tokens).
CHUNK_WORDS = _i("CHUNK_WORDS", 300)
CHUNK_OVERLAP_WORDS = _i("CHUNK_OVERLAP_WORDS", 50)

# ASR confidence. Below REJECT we ask the user to repeat; between REJECT and ACCEPT
# we proceed but show "I heard: ..." so the user can catch mistakes.
ASR_REJECT_LOGPROB = _f("ASR_REJECT_LOGPROB", -1.0)
ASR_ACCEPT_LOGPROB = _f("ASR_ACCEPT_LOGPROB", -0.5)
ASR_NO_SPEECH_REJECT = _f("ASR_NO_SPEECH_REJECT", 0.6)
ASR_MIN_WORDS = _i("ASR_MIN_WORDS", 1)
MAX_AUDIO_SECONDS = _f("MAX_AUDIO_SECONDS", 30)

# Retrieval. Confidence is the best cosine similarity of the dense model (0..1).
TOP_K = _i("TOP_K", 5)
RRF_K = _i("RRF_K", 60)
RETRIEVAL_MIN_CONFIDENCE = _f("RETRIEVAL_MIN_CONFIDENCE", 0.55)

# Groq resilience (decision 5 in CLAUDE.md).
GROQ_RETRIES = _i("GROQ_RETRIES", 2)
GROQ_BACKOFF_S = _f("GROQ_BACKOFF_S", 1.0)
GROQ_BREAKER_FAILURES = _i("GROQ_BREAKER_FAILURES", 3)
GROQ_BREAKER_COOLDOWN_S = _f("GROQ_BREAKER_COOLDOWN_S", 60)

HISTORY_TURNS = _i("HISTORY_TURNS", 3)
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
METRICS_PATH = os.getenv("METRICS_PATH", "logs/latency.jsonl")
