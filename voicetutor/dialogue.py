"""Intent routing and per-session conversation state."""
import re
from dataclasses import dataclass, field

from . import config, llm
from .ingest import Document
from .retrieval import Index

HELP_TEXT = ("You can ask any question about your notes. You can also say: summarise pages 2 to 4, "
             "summarise the section on a topic, read page 3, or read the section on a topic. "
             "Say repeat to hear the last answer again, slower or faster to change my speed, "
             "and shorter or more detail to change how long my answers are.")
NOT_FOUND = "I could not find that in your notes."
NOT_FROM_NOTES = "This is not from your notes."
OFFLINE_NOTE = "Online answers are unavailable right now, so I am reading the closest passages from your notes."

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50})

RULES = [  # (intent, pattern) checked in order; first match wins
    ("help", r"^\s*(help|what can you do|what can i say|commands)\b"),
    ("repeat", r"\b(repeat|say (that|it) again|pardon|come again)\b"),
    ("slower", r"\b(slower|slow down|speak slowly)\b"),
    ("faster", r"\b(faster|speed up|speak quickly)\b"),
    ("shorter", r"\b(shorter|be brief|briefer|less detail)\b"),
    ("longer", r"\b(more detail|more details|longer|elaborate|explain more)\b"),
    ("read", r"^\s*(please\s+)?(read|read out|read aloud)\b"),
    ("summarize", r"\b(summari[sz]e|summary|overview|gist)\b"),
]
QUESTION_START = r"^\s*(what|why|how|when|where|who|which|is|are|does|do|can|explain|define|describe|tell me)\b"
FOLLOW_UP = r"\b(it|its|that|this|they|them|those|these|he|she)\b|^\s*(and|what about|how about)\b"


COMMANDS = {"help", "repeat", "slower", "faster", "shorter", "longer"}


def rule_intent(text: str) -> str | None:
    t = text.lower()
    short = len(t.split()) <= 6  # "why are longer frames used?" is a question, not the 'longer' command
    for intent, pat in RULES:
        if re.search(pat, t) and (short or intent not in COMMANDS):
            return intent
    return None


def route(text: str) -> str:
    """Rules first; if none match and it doesn't look like a question, ask the LLM; default to question."""
    intent = rule_intent(text)
    if intent:
        return intent
    if not re.search(QUESTION_START, text.lower()) and "?" not in text and llm.available():
        try:
            label = llm.classify_intent(text)
            if label in {"question", "summarize", "read", "repeat", "slower", "faster", "shorter", "longer", "help"}:
                return label
        except llm.GroqUnavailable:
            pass
    return "question"


def parse_pages(text: str) -> tuple[int, int] | None:
    """Find 'page 3', 'pages 3 to 5', 'pages three to five', 'p 3-5'. Returns (start, end) or None."""
    t = text.lower()
    for w, n in NUMBER_WORDS.items():
        t = re.sub(rf"\b{w}\b", str(n), t)
    m = re.search(r"\bp(?:age|ages|g)?s?\.?\s*(\d+)(?:\s*(?:-|to|through|and|until)\s*(\d+))?", t)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return (min(a, b), max(a, b))


def section_name(text: str) -> str | None:
    m = re.search(r"\b(?:section|part|chapter|topic)\s+(?:on|about|of|called)?\s*(.+)", text, re.I)
    if m:
        return m.group(1).strip(" ?.")
    m = re.search(r"\b(?:summari[sz]e|read(?: out| aloud)?)\s+(?:the\s+)?(?:notes on|bit about|part about)?\s*(.+)",
                  text, re.I)
    if m and not re.search(r"^(it|that|this|the notes|my notes|everything|all)\b", m.group(1), re.I):
        return m.group(1).strip(" ?.")
    return None


@dataclass
class Reply:
    text: str                                   # shown on screen
    speech: str = ""                            # what TTS reads (defaults to text)
    pages: list[int] = field(default_factory=list)
    intent: str = "question"
    replay: bool = False                        # True: reuse the cached audio, no LLM/TTS
    resynthesize: bool = False                  # True: same speech, new rate

    def __post_init__(self):
        self.speech = self.speech or self.text


@dataclass
class Session:
    """Everything one browser tab owns. Nothing here is shared between users or written to disk."""
    doc: Document | None = None
    index: Index | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    last_question: str = ""
    last_intent_text: str = ""
    last_reply: Reply | None = None
    last_audio: str | None = None
    rate: int = 0                                # edge-tts percent, -50..+50
    length: str = "normal"                       # short | normal | detailed
    general_ok: bool = False

    def handle(self, text: str, timer=None) -> Reply:
        """Route one utterance and produce a reply. Never raises."""
        try:
            reply = self._handle(text.strip(), timer)
        except Exception as e:  # last line of defence: say something useful
            reply = Reply(f"Sorry, something went wrong while answering ({type(e).__name__}). Please try again.")
        if not reply.replay and not reply.resynthesize and reply.intent not in ("help",):
            self.last_reply = reply
        return reply

    def _handle(self, text: str, timer) -> Reply:
        if not text:
            return Reply("I did not catch that, please say it again.", intent="empty")
        intent = route(text)
        if intent == "help":
            return Reply(HELP_TEXT, intent="help")
        if intent == "repeat":
            if not self.last_reply:
                return Reply("There is nothing to repeat yet.", intent="repeat")
            return Reply(self.last_reply.text, self.last_reply.speech, self.last_reply.pages, "repeat", replay=True)
        if intent in ("slower", "faster"):
            self.rate = max(-50, min(50, self.rate + (-15 if intent == "slower" else 15)))
            if not self.last_reply:
                return Reply(f"OK, I will speak {intent}.", intent=intent)
            return Reply(self.last_reply.text, self.last_reply.speech, self.last_reply.pages, intent, resynthesize=True)
        if intent in ("shorter", "longer"):
            self.length = {"shorter": {"detailed": "normal", "normal": "short", "short": "short"},
                           "longer": {"short": "normal", "normal": "detailed", "detailed": "detailed"}}[intent][self.length]
            if not self.last_intent_text:
                return Reply(f"OK, my answers will be {'shorter' if intent == 'shorter' else 'more detailed'}.",
                             intent=intent)
            return self._handle(self.last_intent_text, timer)  # regenerate the last answer at the new length
        self.last_intent_text = text
        if intent == "read":
            return self._read(text)
        if intent == "summarize":
            return self._summarize(text, timer)
        return self._question(text, timer)

    # ---- intents -------------------------------------------------------------------------

    def _need_doc(self) -> Reply | None:
        if self.doc is None:
            return Reply("Please upload your notes first. Use the upload box, then ask again.", intent="no_doc")
        return None

    def _read(self, text: str) -> Reply:
        if (r := self._need_doc()):
            return r
        rng = parse_pages(text)
        if rng:
            a, b = rng
            if a < 1 or a > self.doc.num_pages:
                return Reply(f"Your notes have {self.doc.num_pages} pages, so there is no page {a}.", intent="read")
            b = min(b, a + 1, self.doc.num_pages)  # read at most two pages at a time
            parts = []
            for p in range(a, b + 1):
                body = " ".join(self.doc.pages[p - 1].split())
                parts.append(f"Page {p}. " + (body if p not in self.doc.unreadable_pages
                                              else "This page has no readable text. It may be an image."))
            return Reply(" ".join(parts), pages=list(range(a, b + 1)), intent="read")
        name = section_name(text)
        if not name:
            return Reply("Which page or section should I read? For example, say read page 2.", intent="read")
        chunks = self._find_section(name, n=3)
        if not chunks:
            return Reply(f"I could not find a section called {name} in your notes.", intent="read")
        pages = sorted({c.page for c in chunks})
        return Reply(f"From page {pages[0]}. " + " ".join(c.text for c in chunks), pages=pages, intent="read")

    def _find_section(self, name: str, n: int):
        hits, conf = self.index.search(name, k=1)
        if not hits or conf < config.RETRIEVAL_MIN_CONFIDENCE:
            return []
        return self.index.neighbours(hits[0].chunk, n)

    def _summarize(self, text: str, timer) -> Reply:
        if (r := self._need_doc()):
            return r
        rng = parse_pages(text)
        if rng:
            a, b = max(1, rng[0]), min(rng[1], self.doc.num_pages)
            if a > b:
                return Reply(f"Your notes have {self.doc.num_pages} pages.", intent="summarize")
            chunks = [c for c in self.doc.chunks if a <= c.page <= b]
        elif (name := section_name(text)):
            chunks = self._find_section(name, n=4)
            if not chunks:
                return Reply(f"I could not find a section called {name} in your notes.", intent="summarize")
        else:
            chunks = self.doc.chunks
        chunks = _first_per_page(chunks)[:12]
        pages = sorted({c.page for c in chunks})
        if not chunks:
            return Reply("Those pages have no readable text.", intent="summarize")
        try:
            with _stage(timer, "llm"):
                return Reply(llm.summarize(chunks, self.length), pages=pages, intent="summarize")
        except llm.GroqUnavailable as e:
            return Reply(_offline_prefix(e) + _extractive(chunks, self.length), pages=pages, intent="summarize")

    def _question(self, text: str, timer) -> Reply:
        query = text
        if self.history and re.search(FOLLOW_UP, text.lower()) and llm.available():
            try:
                with _stage(timer, "rewrite"):
                    query = llm.rewrite_query(text, self.history[-config.HISTORY_TURNS:])
            except llm.GroqUnavailable:
                pass
        self.last_question = query
        if self.doc is None:
            if self.general_ok:
                return self._general(query, timer)
            return self._need_doc()
        with _stage(timer, "retrieval"):
            hits, conf = self.index.search(query)
        if conf < config.RETRIEVAL_MIN_CONFIDENCE:
            return self._general(query, timer) if self.general_ok else Reply(NOT_FOUND, intent="not_found")
        chunks = [h.chunk for h in hits]
        try:
            with _stage(timer, "llm"):
                answer = llm.grounded_answer(query, chunks, self.length)
        except llm.GroqUnavailable as e:
            confident = [h.chunk for h in hits if h.cosine >= config.RETRIEVAL_MIN_CONFIDENCE]
            top = confident[:{"short": 1, "normal": 2, "detailed": 3}[self.length]]
            reply = Reply(_offline_prefix(e) + _extractive(top, self.length), pages=sorted({c.page for c in top}))
            self.history.append((query, reply.text))
            return reply
        if "NOT_FOUND" in answer:
            return self._general(query, timer) if self.general_ok else Reply(NOT_FOUND, intent="not_found")
        pages = sorted({c.page for c in chunks if re.search(rf"\bpage {c.page}\b", answer)}) \
            or sorted({c.page for c in chunks[:3]})
        self.history.append((query, answer))
        return Reply(answer, pages=pages)

    def _general(self, query: str, timer) -> Reply:
        try:
            with _stage(timer, "llm"):
                answer = llm.general_answer(query, self.length)
        except llm.GroqUnavailable:
            return Reply(NOT_FOUND + " General knowledge answers need the online model, which is unavailable.",
                         intent="not_found")
        self.history.append((query, answer))
        return Reply(f"{NOT_FROM_NOTES} {answer}", intent="general")


def _offline_prefix(err: Exception) -> str:
    # With no key at all the extractive mode is normal operation, so we don't apologise every turn.
    return "" if str(err) == "no key" else OFFLINE_NOTE + " "


def _first_per_page(chunks):
    seen, out = set(), []
    for c in chunks:
        if c.page not in seen:
            seen.add(c.page)
            out.append(c)
    return out


def _extractive(chunks, length: str) -> str:
    """No-LLM answer: the opening sentences of each passage, prefixed with its page."""
    limit = {"short": 2, "normal": 3, "detailed": 5}[length]
    parts = []
    for c in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", c.text)
        parts.append(f"From page {c.page}: " + " ".join(sentences[:limit]))
    return " ".join(parts)


class _stage:
    """Use timer.stage(name) if a timer was given, else do nothing."""

    def __init__(self, timer, name):
        self.cm = timer.stage(name) if timer else None

    def __enter__(self):
        return self.cm.__enter__() if self.cm else None

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc) if self.cm else False
