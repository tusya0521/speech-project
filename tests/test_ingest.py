import pytest

from voicetutor import config
from voicetutor.ingest import IngestError, chunk_pages, ingest, load_pages, top_keywords


def test_chunks_keep_page_and_respect_size():
    pages = ["word " * 700, "short page text here"]
    chunks = chunk_pages(pages, size=300, overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text.split()) <= 350 for c in chunks)       # size + overlap
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2 and chunks[-1].text == "short page text here"


def test_overlap_repeats_tail_of_previous_chunk():
    words = [f"w{i}" for i in range(600)]
    chunks = chunk_pages([" ".join(words)], size=300, overlap=50)
    assert chunks[1].text.split()[:50] == words[250:300]


def test_paragraphs_are_not_split_when_they_fit():
    page = "alpha beta gamma.\n\ndelta epsilon.\n\nzeta eta theta."
    chunks = chunk_pages([page], size=4, overlap=0)
    assert [c.text for c in chunks] == ["alpha beta gamma.", "delta epsilon.", "zeta eta theta."]


def test_no_chunk_crosses_a_page():
    chunks = chunk_pages(["a b c", "d e f"], size=100, overlap=10)
    assert [(c.page, c.text) for c in chunks] == [(1, "a b c"), (2, "d e f")]


def test_txt_and_unreadable_pages(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Hidden Markov models are used.\fx\fViterbi decoding finds the best path.", encoding="utf-8")
    doc = ingest(str(p))
    assert doc.num_pages == 3 and doc.unreadable_pages == [2]


@pytest.mark.parametrize("name,content,msg", [
    ("bad.pdf", b"not a pdf", "could not open"),
    ("x.exe", b"MZ", "only read"),
    ("empty.txt", b"   ", "readable text"),
])
def test_bad_files_raise_friendly_errors(tmp_path, name, content, msg):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(IngestError, match=msg):
        load_pages(str(p))


def test_missing_and_oversized(tmp_path, monkeypatch):
    with pytest.raises(IngestError):
        load_pages(None)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.000001)
    p = tmp_path / "a.txt"
    p.write_text("some words here", encoding="utf-8")
    with pytest.raises(IngestError, match="limit"):
        load_pages(str(p))


def test_keywords_prefer_distinctive_terms():
    pages = ["the model uses mfcc features. mfcc features matter.",
             "the model uses beam search. beam search prunes.",
             "the model is trained."]
    kws = top_keywords(pages, 5)
    assert "model" not in kws  # appears on every page, so idf is zero
    assert any("mfcc" in k for k in kws) and any("beam" in k for k in kws)
