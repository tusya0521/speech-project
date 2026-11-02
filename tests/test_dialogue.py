import pytest

from voicetutor import config, llm
from voicetutor.dialogue import NOT_FOUND, NOT_FROM_NOTES, OFFLINE_NOTE, Session, parse_pages, route, section_name
from voicetutor.ingest import Document, chunk_pages
from voicetutor.retrieval import Index

PAGES = ["Hidden Markov models have hidden states and transition probabilities. The Viterbi algorithm decodes.",
         "MFCC features use a mel filterbank, a logarithm and the discrete cosine transform.",
         "",
         "Word error rate counts substitutions, deletions and insertions over reference words."]


@pytest.fixture
def session():
    doc = Document("Notes", PAGES, chunk_pages(PAGES), unreadable_pages=[3], keywords=["mfcc"])
    return Session(doc=doc, index=Index(doc.chunks))


@pytest.mark.parametrize("text,intent", [
    ("help", "help"), ("repeat that", "repeat"), ("say it again", "repeat"),
    ("slower please", "slower"), ("speak faster", "faster"), ("shorter", "shorter"),
    ("more detail please", "longer"), ("read page 2", "read"), ("summarize pages 1 to 3", "summarize"),
    ("give me an overview of hidden markov models", "summarize"),
    ("what is a mel filterbank?", "question"),
    ("why are longer frames used in speech processing systems?", "question"),  # not the 'longer' command
])
def test_rule_routing(text, intent):
    assert route(text) == intent


@pytest.mark.parametrize("text,rng", [
    ("read page 12", (12, 12)), ("summarize pages 3 to 5", (3, 5)), ("pages 5-3", (3, 5)),
    ("read page three", (3, 3)), ("summarise pages two through four", (2, 4)), ("what is wer", None),
])
def test_parse_pages(text, rng):
    assert parse_pages(text) == rng


def test_section_name():
    assert section_name("read the section on hidden markov models") == "hidden markov models"
    assert section_name("summarize mfcc features") == "mfcc features"
    assert section_name("summarize it") is None


def test_question_offline_is_extractive_with_page(session):
    r = session.handle("How is word error rate counted?")
    assert r.text.startswith("From page 4") and r.pages == [4] and OFFLINE_NOTE not in r.text


def test_out_of_scope_is_refused(session, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_CONFIDENCE", 0.9)
    r = session.handle("Who won the world cup?")
    assert r.text == NOT_FOUND and r.intent == "not_found"


def test_general_knowledge_is_labelled(session, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_CONFIDENCE", 0.9)
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(llm, "general_answer", lambda q, l: "Canberra.")
    session.general_ok = True
    r = session.handle("What is the capital of Australia?")
    assert r.text.startswith(NOT_FROM_NOTES) and r.intent == "general"


def test_grounded_answer_and_not_found_token(session, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(llm, "grounded_answer", lambda q, c, l: "On page 2, MFCCs use a mel filterbank.")
    r = session.handle("What do MFCC features use?")
    assert r.pages == [2]
    monkeypatch.setattr(llm, "grounded_answer", lambda q, c, l: "NOT_FOUND")
    assert session.handle("What do MFCC features use?").text == NOT_FOUND


def test_groq_failure_falls_back_with_note(session, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")

    def boom(*a):
        raise llm.GroqUnavailable("rate limited")
    monkeypatch.setattr(llm, "grounded_answer", boom)
    r = session.handle("How is word error rate counted?")
    assert r.text.startswith(OFFLINE_NOTE) and "page 4" in r.text


def test_repeat_does_not_call_llm(session, monkeypatch):
    first = session.handle("How is word error rate counted?")
    session.last_audio = "cached.mp3"
    monkeypatch.setattr(llm, "grounded_answer", lambda *a: pytest.fail("repeat must not call the LLM"))
    monkeypatch.setattr(session.index, "search", lambda *a, **k: pytest.fail("repeat must not search"))
    r = session.handle("repeat")
    assert r.replay and r.text == first.text


def test_slower_resynthesizes_and_changes_rate(session):
    session.handle("How is word error rate counted?")
    r = session.handle("slower")
    assert r.resynthesize and session.rate == -15


def test_shorter_regenerates_last_answer(session):
    session.handle("How is word error rate counted?")
    session.handle("more detail")
    assert session.length == "detailed"


def test_follow_up_is_rewritten(session, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    seen = {}
    monkeypatch.setattr(llm, "grounded_answer", lambda q, c, l: seen.setdefault("q", q) and "On page 1, yes.")
    monkeypatch.setattr(llm, "rewrite_query", lambda q, h: "What does the Viterbi algorithm do in HMMs?")
    session.handle("What are hidden Markov models?")
    seen.clear()
    session.handle("What does it decode with?")
    assert seen["q"] == "What does the Viterbi algorithm do in HMMs?"


def test_read_page_and_unreadable_page(session):
    assert session.handle("read page 2").text.startswith("Page 2. MFCC")
    assert "no readable text" in session.handle("read page 3").text
    assert "no page 9" in session.handle("read page 9").text


def test_summarize_offline_uses_page_range(session):
    r = session.handle("summarize pages 1 to 2")
    assert r.pages == [1, 2] and "From page 1" in r.text


def test_no_document_and_empty_input():
    s = Session()
    assert "upload" in s.handle("What is MFCC?").text.lower()
    assert "did not catch" in s.handle("   ").text


def test_handle_never_raises(session, monkeypatch):
    monkeypatch.setattr(session.index, "search", lambda *a, **k: 1 / 0)
    assert "went wrong" in session.handle("What is MFCC?").text
