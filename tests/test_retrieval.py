from voicetutor.ingest import Chunk
from voicetutor.retrieval import Index, rrf

CHUNKS = [Chunk("Hidden Markov models have states and transition probabilities.", 1, 0),
          Chunk("MFCC features use a mel filterbank and the cosine transform.", 2, 1),
          Chunk("Word error rate counts substitutions deletions and insertions.", 3, 2),
          Chunk("More about word error rate normalisation.", 4, 3)]


def test_rrf_rewards_agreement():
    fused = dict(rrf([[0, 1, 2], [1, 0, 2]], k=60))
    assert fused[0] == fused[1] > fused[2]
    assert abs(fused[2] - 2 / 63) < 1e-12


def test_search_finds_right_page_and_reports_confidence():
    hits, conf = Index(CHUNKS).search("how is word error rate computed from substitutions")
    assert hits[0].chunk.page == 3
    assert 0 < conf <= 1 and conf == max(h.cosine for h in hits)


def test_unrelated_query_has_low_confidence():
    _, conf = Index(CHUNKS).search("chocolate cake recipe")
    assert conf < 0.2


def test_hyphenated_asr_spelling_matches_joined_term():
    ix = Index([Chunk("A triphone models context.", 1, 0), Chunk("Beam search prunes.", 2, 1)])
    hits, conf = ix.search("what is a tri-phone")
    assert hits[0].chunk.page == 1 and conf > 0.3


def test_empty_query_and_neighbours():
    ix = Index(CHUNKS)
    assert ix.search("   ") == ([], 0.0)
    assert [c.page for c in ix.neighbours(CHUNKS[2], 3)] == [3, 4]
