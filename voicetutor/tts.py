"""Text-to-speech: edge-tts first, gTTS then pyttsx3 as fallbacks. speech_prep() makes text listenable."""
import asyncio
import re
import tempfile

from . import config

SYMBOLS = [  # order matters: multi-character symbols first
    ("<=", " less than or equal to "), (">=", " greater than or equal to "), ("!=", " not equal to "),
    ("≤", " less than or equal to "), ("≥", " greater than or equal to "), ("≠", " not equal to "),
    ("≈", " approximately "), ("->", " leads to "), ("→", " leads to "), ("×", " times "),
    ("÷", " divided by "), ("±", " plus or minus "), ("%", " percent"), ("&", " and "),
    ("=", " equals "), ("+", " plus "), ("<", " less than "), (">", " greater than "),
    ("√", " square root of "), ("∑", " sum of "), ("Σ", " sum of "), ("λ", " lambda "),
    ("μ", " mu "), ("σ", " sigma "), ("θ", " theta "), ("α", " alpha "), ("β", " beta "), ("π", " pi "),
]
ABBREVIATIONS = [(r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\betc\.", "and so on"),
                 (r"\bvs\.?", "versus"), (r"\bpp\.\s*", "pages "), (r"\bp\.\s*(?=\d)", "page ")]
MAX_SENTENCE_WORDS = 25


def speech_prep(text: str) -> str:
    """Convert LLM/markdown text into short plain sentences that a TTS voice reads well."""
    if not text:
        return ""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)                  # code blocks
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)                  # links/images -> label
    t = re.sub(r"https?://\S+", "a web link", t)
    lines = []
    for line in t.splitlines():
        line = line.strip()
        if re.fullmatch(r"\|?[\s:|-]+\|?", line) and "-" in line:   # table separator row
            continue
        if line.count("|") >= 2:                                       # table row -> comma list
            line = ", ".join(c.strip() for c in line.strip("|").split("|") if c.strip())
        line = re.sub(r"^#{1,6}\s*", "", line)                         # headings
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line)              # bullets / numbered lists
        if line:
            lines.append(line if re.search(r"[.!?:]$", line) else line + ".")
    t = " ".join(lines)
    t = re.sub(r"(\*\*|__|\*|_|`|~~)", "", t)                          # emphasis marks
    for pat, rep in ABBREVIATIONS:
        t = re.sub(pat, rep, t, flags=re.I)
    t = re.sub(r"(\w)\^2\b", r"\1 squared", t)
    t = re.sub(r"(\w)\^3\b", r"\1 cubed", t)
    t = re.sub(r"(\w)\^(\w+)", r"\1 to the power of \2", t)
    t = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", t)                     # ranges: 3-5
    t = re.sub(r"(\d)\s*/\s*(\d)|\s/\s", lambda m: f"{m.group(1)} over {m.group(2)}" if m.group(1) else " over ", t)
    for sym, word in SYMBOLS:
        t = t.replace(sym, word)
    t = re.sub(r"[;:]\s+", ". ", t)                                    # break clauses into sentences
    t = re.sub(r"[()\[\]{}\"#*|\\]", " ", t)
    t = re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s+", " ", t)).strip()
    t = re.sub(r"\.{2,}", ".", t)
    return " ".join(_split_long(s) for s in re.split(r"(?<=[.!?])\s+", t) if s)


def _split_long(sentence: str) -> str:
    """Split a sentence longer than MAX_SENTENCE_WORDS at the comma nearest its middle."""
    words = sentence.split()
    if len(words) <= MAX_SENTENCE_WORDS:
        return sentence
    commas = [i for i, w in enumerate(words[:-1]) if w.endswith(",")]
    if not commas:
        return sentence
    cut = min(commas, key=lambda i: abs(i - len(words) // 2))
    first = " ".join(words[:cut + 1]).rstrip(",") + "."
    rest = " ".join(words[cut + 1:])
    return _split_long(first) + " " + _split_long(rest[:1].upper() + rest[1:])


def rate_string(rate: int) -> str:
    """edge-tts rate, e.g. -20 -> '-20%'."""
    return f"{'+' if rate >= 0 else ''}{int(rate)}%"


def synthesize(text: str, rate: int = 0) -> tuple[str | None, str]:
    """Speak text to an audio file. Returns (path or None, backend name). Never raises."""
    text = text.strip()
    if not text:
        return None, "none"
    for name, fn in (("edge-tts", _edge), ("gtts", _gtts), ("pyttsx3", _pyttsx3)):
        try:
            return fn(text, rate), name
        except Exception:
            continue
    return None, "none"


def _edge(text: str, rate: int) -> str:
    import edge_tts
    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    asyncio.run(edge_tts.Communicate(text, config.TTS_VOICE, rate=rate_string(rate)).save(path))
    return path


def _gtts(text: str, rate: int) -> str:
    from gtts import gTTS
    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    gTTS(text, lang="en", slow=rate <= -30).save(path)
    return path


def _pyttsx3(text: str, rate: int) -> str:
    import pyttsx3  # offline; needs espeak on Linux, SAPI5 on Windows
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    engine = pyttsx3.init()
    engine.setProperty("rate", int(175 * (1 + rate / 100)))
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path
