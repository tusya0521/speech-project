"""Load PDF/DOCX/TXT into pages, then split pages into page-aware chunks."""
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from . import config

WORDS_PER_PSEUDO_PAGE = 400  # DOCX/TXT have no pages, so we number fixed-size parts instead

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being
this that these those it its into than then there their they them we you your our he she his her which who
whom what when where why how all any both each few more most other some such no nor not only own same so too
very can will just should now also may might must shall would could do does did done have has had having i
me my one two use used using page pages figure table e g i e etc via per within between over under about""".split())


class IngestError(Exception):
    """Raised with a user-facing message when a file cannot be loaded."""


@dataclass
class Chunk:
    text: str
    page: int
    idx: int


@dataclass
class Document:
    title: str
    pages: list[str]                       # pages[0] is page 1
    chunks: list[Chunk]
    unreadable_pages: list[int] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def load_pages(path: str) -> tuple[str, list[str]]:
    """Return (title, list of page texts). Raises IngestError on any problem."""
    if not path or not os.path.exists(path):
        raise IngestError("I did not receive a file. Please upload a PDF, Word or text file.")
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > config.MAX_UPLOAD_MB:
        raise IngestError(f"That file is {size_mb:.0f} megabytes. The limit is {config.MAX_UPLOAD_MB:.0f}.")
    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
    try:
        if ext == ".pdf":
            title, pages = _load_pdf(path, name)
        elif ext == ".docx":
            import docx
            text = "\n\n".join(p.text for p in docx.Document(path).paragraphs)
            title, pages = name, _pseudo_pages(text)
        elif ext in (".txt", ".md"):
            with open(path, encoding="utf-8", errors="replace") as f:
                title, pages = name, _pseudo_pages(f.read())
        else:
            raise IngestError("I can only read PDF, Word (.docx) or text files.")
    except IngestError:
        raise
    except Exception:
        raise IngestError("I could not open that file. It may be damaged or password protected.")
    if len(pages) > config.MAX_PAGES:
        raise IngestError(f"That document has {len(pages)} pages. The limit is {config.MAX_PAGES}.")
    if not any(p.strip() for p in pages):
        raise IngestError("I could not find any readable text. The file may be scanned images.")
    return title, pages


def _load_pdf(path: str, fallback_title: str) -> tuple[str, list[str]]:
    import pymupdf
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise IngestError("That PDF is password protected.")
        title = (pdf.metadata or {}).get("title") or fallback_title
        return title, [page.get_text("text") for page in pdf]


def _pseudo_pages(text: str) -> list[str]:
    if "\f" in text:
        return text.split("\f")
    words = text.split()
    if not words:
        return [""]
    n = math.ceil(len(words) / WORDS_PER_PSEUDO_PAGE)
    # Re-split on paragraphs so we don't cut mid-paragraph more than needed.
    paras, pages, cur, count = re.split(r"\n\s*\n", text), [], [], 0
    for p in paras:
        cur.append(p)
        count += len(p.split())
        if count >= WORDS_PER_PSEUDO_PAGE and len(pages) < n - 1:
            pages.append("\n\n".join(cur))
            cur, count = [], 0
    pages.append("\n\n".join(cur))
    return pages


def chunk_pages(pages: list[str], size: int = None, overlap: int = None) -> list[Chunk]:
    """Paragraph-based chunks of about `size` words, never crossing a page boundary.

    Consecutive chunks on the same page share `overlap` words so an answer that
    straddles a split is still retrievable.
    """
    size = size or config.CHUNK_WORDS
    overlap = config.CHUNK_OVERLAP_WORDS if overlap is None else overlap
    chunks: list[Chunk] = []
    for page_no, text in enumerate(pages, start=1):
        # 1. Paragraphs as word lists; oversized paragraphs are cut into size-word pieces.
        pieces: list[list[str]] = []
        for para in re.split(r"\n\s*\n", text):
            words = para.split()
            pieces += [words[i:i + size] for i in range(0, len(words), size)]
        # 2. Greedily pack whole pieces into groups of at most `size` words.
        groups: list[list[str]] = []
        for words in pieces:
            if groups and len(groups[-1]) + len(words) <= size:
                groups[-1] += words
            else:
                groups.append(list(words))
        # 3. Prefix each group with the tail of the previous one on this page (overlap).
        for g, words in enumerate(groups):
            prefix = groups[g - 1][-overlap:] if g > 0 and overlap else []
            chunks.append(Chunk(" ".join(prefix + words), page_no, len(chunks)))
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def top_keywords(pages: list[str], n: int = 25) -> list[str]:
    """TF-IDF over pages (each page = one 'document'); returns the n highest-scoring terms.

    Bigrams are included so phrases like 'hidden markov' survive. Used for Whisper
    vocabulary biasing and for the spoken 'main topics' summary.
    """
    docs = []
    for p in pages:
        toks = [t for t in tokenize(p) if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]
        docs.append(toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])])
    df = Counter(term for d in docs for term in set(d))
    total = Counter(term for d in docs for term in d)
    n_docs = len(docs)
    scores: Counter = Counter()
    for d in docs:
        for term, tf in Counter(d).items():
            if total[term] < 2:
                continue  # one-off words are usually noise
            idf = math.log(n_docs / df[term]) if n_docs > 1 else 1.0  # 0 for words on every page
            scores[term] = max(scores[term], tf * idf * (1.5 if " " in term else 1.0))
    out: list[str] = []
    for term, _ in scores.most_common():
        if any(term in o or o in term for o in out):
            continue  # skip 'markov' if 'hidden markov' is already chosen, and vice versa
        out.append(term)
        if len(out) == n:
            break
    return out


def ingest(path: str) -> Document:
    title, pages = load_pages(path)
    unreadable = [i for i, p in enumerate(pages, start=1) if len(p.split()) < 3]
    return Document(title=title.strip(), pages=pages, chunks=chunk_pages(pages),
                    unreadable_pages=unreadable, keywords=top_keywords(pages))
