"""Groq client with retry/backoff and a circuit breaker, plus the prompts VoiceTutor uses."""
import time

from . import config


class GroqUnavailable(Exception):
    """No key, breaker open, or all retries failed. Callers fall back to offline behaviour."""


class Breaker:
    """After N consecutive failures, skip Groq for a cooldown period (shared by ASR and LLM)."""

    def __init__(self, failures: int, cooldown_s: float, clock=time.monotonic):
        self.limit, self.cooldown, self.clock = failures, cooldown_s, clock
        self.fails, self.open_until = 0, 0.0

    def allow(self) -> bool:
        return self.clock() >= self.open_until

    def success(self):
        self.fails = 0

    def failure(self):
        self.fails += 1
        if self.fails >= self.limit:
            self.open_until, self.fails = self.clock() + self.cooldown, 0


breaker = Breaker(config.GROQ_BREAKER_FAILURES, config.GROQ_BREAKER_COOLDOWN_S)
_client = None


def client():
    global _client
    if _client is None:
        from groq import Groq
        # Retries are handled here so the backoff is visible and testable.
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=0, timeout=20)
    return _client


def available() -> bool:
    return bool(config.GROQ_API_KEY) and breaker.allow()


def call_groq(fn, *args, **kwargs):
    """Run fn with retries and exponential backoff; raise GroqUnavailable if it never succeeds."""
    if not available():
        raise GroqUnavailable("no key" if not config.GROQ_API_KEY else "cooling down")
    for attempt in range(config.GROQ_RETRIES + 1):
        try:
            out = fn(*args, **kwargs)
            breaker.success()
            return out
        except Exception as e:  # network, rate limit, 5xx: all treated the same
            last = e
            if attempt < config.GROQ_RETRIES:
                time.sleep(config.GROQ_BACKOFF_S * 2 ** attempt)
    breaker.failure()
    raise GroqUnavailable(str(last))


def chat(messages: list[dict], model: str = None, max_tokens: int = 400, temperature: float = 0.2) -> str:
    resp = call_groq(lambda: client().chat.completions.create(
        model=model or config.GROQ_LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature))
    return resp.choices[0].message.content.strip()


# ---- prompts -------------------------------------------------------------------------------

LENGTH_HINT = {"short": "Answer in 1 to 2 sentences.",
               "normal": "Answer in 3 to 5 sentences.",
               "detailed": "Answer in up to 8 sentences with more explanation."}

GROUNDED_SYSTEM = """You are VoiceTutor, a study assistant whose answers are read aloud to visually impaired and dyslexic students.
Answer ONLY from the numbered passages below. Each passage starts with its page number.
Mention the page where you found the answer, in words like "on page 12".
If the passages do not contain the answer, reply exactly: NOT_FOUND
Write plain spoken English: no markdown, no bullet points, no tables, short sentences. {length}"""

GENERAL_SYSTEM = """You are VoiceTutor, a study assistant whose answers are read aloud.
The student's notes did not cover this question, and they allowed general knowledge answers.
Answer from general knowledge in plain spoken English, no markdown, short sentences. {length}"""

SUMMARY_SYSTEM = """You are VoiceTutor. Summarise the passages below for a student who will listen to it.
Mention page numbers in words like "on page 3". Plain spoken English, no markdown, short sentences. {length}"""


def format_passages(chunks) -> str:
    return "\n\n".join(f"[{i}] (page {c.page}) {c.text}" for i, c in enumerate(chunks, start=1))


def grounded_answer(question: str, chunks, length: str = "normal") -> str:
    return chat([{"role": "system", "content": GROUNDED_SYSTEM.format(length=LENGTH_HINT[length])},
                 {"role": "user", "content": f"Passages:\n{format_passages(chunks)}\n\nQuestion: {question}"}])


def general_answer(question: str, length: str = "normal") -> str:
    return chat([{"role": "system", "content": GENERAL_SYSTEM.format(length=LENGTH_HINT[length])},
                 {"role": "user", "content": question}])


def summarize(chunks, length: str = "normal") -> str:
    return chat([{"role": "system", "content": SUMMARY_SYSTEM.format(length=LENGTH_HINT[length])},
                 {"role": "user", "content": format_passages(chunks)}], max_tokens=500)


def rewrite_query(question: str, history: list[tuple[str, str]]) -> str:
    """Turn a follow-up ("what about its drawbacks?") into a standalone question."""
    turns = "\n".join(f"Student: {q}\nTutor: {a}" for q, a in history)
    return chat([{"role": "system", "content": "Rewrite the student's last question so it makes sense on its own, "
                  "using the conversation for context. Reply with the rewritten question only."},
                 {"role": "user", "content": f"{turns}\n\nLast question: {question}"}],
                model=config.GROQ_FAST_MODEL, max_tokens=80, temperature=0.0)


def classify_intent(text: str) -> str:
    """LLM fallback for the rule router. Returns one intent label."""
    label = chat([{"role": "system", "content": "Classify the student's request as exactly one word: "
                   "question, summarize, read, repeat, slower, faster, shorter, longer, help."},
                  {"role": "user", "content": text}],
                 model=config.GROQ_FAST_MODEL, max_tokens=5, temperature=0.0)
    return label.lower().strip(" .")
