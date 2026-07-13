"""Tests for the n-gram (prompt-lookup) speculator used as the MTP fallback.

These run without MLX/model weights — NGramSpeculator is pure-Python token
history. Imported from exo.worker.engines.mlx.mtp (NOT the old
speculative_generate location, which was never committed).
"""
from exo.worker.engines.mlx.mtp import NGramSpeculator


def test_ngram_proposes_following_tokens():
    spec = NGramSpeculator(ngram_size=3, max_draft=4)
    spec.add_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

    assert spec.propose_drafts([10, 1, 2, 3]) == [4, 5, 1, 2]
    assert spec.propose_drafts([10, 4, 5, 1]) == [2, 3, 4, 5]


def test_ngram_returns_empty_when_no_match():
    spec = NGramSpeculator(ngram_size=3, max_draft=4)
    spec.add_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    assert spec.propose_drafts([99, 98, 97]) == []


def test_ngram_returns_empty_when_recent_too_short():
    spec = NGramSpeculator(ngram_size=3, max_draft=4)
    spec.add_tokens([1, 2, 3, 4, 5])
    assert spec.propose_drafts([1]) == []


def test_ngram_reset_clears_history():
    spec = NGramSpeculator(ngram_size=3, max_draft=4)
    spec.add_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    # [0, 1, 2, 3] -> needle [1,2,3] (last 3) -> first occurrence at history idx 0
    # -> continuation [4,5,1,2].
    assert spec.propose_drafts([0, 1, 2, 3]) == [4, 5, 1, 2]
    spec.reset()
    assert spec.propose_drafts([0, 1, 2, 3]) == []


if __name__ == "__main__":
    test_ngram_proposes_following_tokens()
    test_ngram_returns_empty_when_no_match()
    test_ngram_returns_empty_when_recent_too_short()
    test_ngram_reset_clears_history()
    print("NGramSpeculator tests passed!")
