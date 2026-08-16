"""
test_scorer.py
----------------
Automated unit tests for the hybrid scoring functions in scorer.py.
Unlike test_prototype.py (which prints output for manual inspection),
these use real assertions and a pass/fail exit code -- suitable for a
CI pipeline or a pre-submission check.

Run with:
    python -m pytest tests/test_scorer.py -v

"""
import itertools
import pytest
from data import TRACKS, TRACKS_BY_ID
from scorer import recommend_next_track, audio_similarity, explain, hybrid_score

from scorer import (
    audio_similarity, tag_overlap, sequence_score, context_adjustment,
    hybrid_score, novelty_rerank, recommend_next_track,
)
from data import TRACKS


def test_audio_similarity_identical_track_is_1():
    t = TRACKS[0]
    assert audio_similarity(t, t) == pytest.approx(1.0)


def test_audio_similarity_is_symmetric():
    a, b = TRACKS[0], TRACKS[1]
    assert audio_similarity(a, b) == pytest.approx(audio_similarity(b, a))


def test_audio_similarity_bounded_0_1():
    for a in TRACKS[:20]:
        for b in TRACKS[:20]:
            assert 0.0 <= audio_similarity(a, b) <= 1.0


def test_tag_overlap_empty_tags_is_zero():
    assert tag_overlap({"tags": set()}, {"tags": {"rock"}}) == 0.0


def test_tag_overlap_identical_sets_is_one():
    a = {"tags": {"rock", "indie"}}
    b = {"tags": {"rock", "indie"}}
    assert tag_overlap(a, b) == 1.0


def test_tag_overlap_disjoint_sets_is_zero():
    assert tag_overlap({"tags": {"rock"}}, {"tags": {"jazz"}}) == 0.0


def test_sequence_score_unknown_pair_is_zero():
    assert sequence_score("does-not-exist-a", "does-not-exist-b") == 0.0


def test_context_adjustment_bounded_0_1():
    for track in TRACKS[:20]:
        for mood in ["energetic", "happy", "calm", "melancholy", "neutral"]:
            assert 0.0 <= context_adjustment(track, mood, 0.5) <= 1.0


def test_hybrid_score_bounded_0_1():
    seed_ids = [TRACKS[0]["id"]]
    for candidate in TRACKS[1:30]:
        result = hybrid_score(seed_ids, candidate, "neutral", 0.5)
        assert 0.0 <= result["score"] <= 1.0


def test_novelty_zero_keeps_original_scores():
    # With novelty=0.0, the boost term should vanish entirely, so the
    # adjusted score should exactly equal the base hybrid score.
    seed_ids = [TRACKS[0]["id"]]
    scored = [
        {"id": c["id"], **hybrid_score(seed_ids, c, "neutral", 0.5)}
        for c in TRACKS[1:30]
    ]
    reranked = novelty_rerank(scored, novelty=0.0)
    for item in reranked:
        assert item["adjusted_score"] == pytest.approx(item["score"])


def test_recommend_next_track_excludes_seeds():
    seed_ids = [TRACKS[0]["id"], TRACKS[1]["id"]]
    results = recommend_next_track(seed_ids, top_n=10)
    result_ids = {r["track_id"] for r in results}
    assert not set(seed_ids) & result_ids


def test_recommend_next_track_respects_top_n():
    results = recommend_next_track([TRACKS[0]["id"]], top_n=3)
    assert len(results) == 3


def test_recommend_next_track_scores_in_range():
    results = recommend_next_track([TRACKS[0]["id"]], top_n=5)
    for r in results:
        assert 0.0 <= r["confidence_score"] <= 1.0


def test_recommend_next_track_different_seeds_give_different_top_pick():
    # Sanity check that the system isn't just returning the same track
    # regardless of input -- a rock seed and a classical seed should
    # not recommend the same next track.
    rock_seed = [t["id"] for t in TRACKS if t.get("genre_hint") == "rock"][0]
    classical_seed = [t["id"] for t in TRACKS if t.get("genre_hint") == "classical"][0]
    rock_top = recommend_next_track([rock_seed], top_n=1)[0]["track_id"]
    classical_top = recommend_next_track([classical_seed], top_n=1)[0]["track_id"]
    assert rock_top != classical_top


def test_multi_seed_history_changes_score_vs_last_track_only():
    """Locks in that using the full history changes the score used for
    ranking, rather than silently degenerating back to last-track-only
    scoring. Deterministic (unlike comparing final track_ids, which
    could coincidentally match), so it won't flake."""
    rock_seed = [t["id"] for t in TRACKS if t.get("genre_hint") == "rock"][0]
    classical_seed = [t["id"] for t in TRACKS if t.get("genre_hint") == "classical"][0]
    candidate = TRACKS_BY_ID[
        [t["id"] for t in TRACKS if t["id"] not in (rock_seed, classical_seed)][0]
    ]

    with_history = hybrid_score([rock_seed, classical_seed], candidate, "neutral", 0.5)
    last_only = hybrid_score([classical_seed], candidate, "neutral", 0.5)

    assert with_history["audio"] != pytest.approx(last_only["audio"])


def test_mmr_reduces_pairwise_similarity_vs_naive_top_n():
    """Locks in that MMR (top_n>1, diversity>0) returns a more diverse
    set than a naive top-N-by-score slice (diversity=0), for a
    representative seed. Prevents a future edit from silently
    disabling MMR."""
    seed = TRACKS[0]["id"]

    mmr_results = recommend_next_track([seed], top_n=5, diversity=0.6)
    mmr_tracks = [TRACKS_BY_ID[r["track_id"]] for r in mmr_results]
    mmr_pairs = list(itertools.combinations(mmr_tracks, 2))
    mmr_avg_sim = sum(audio_similarity(a, b) for a, b in mmr_pairs) / len(mmr_pairs)

    naive_results = recommend_next_track([seed], top_n=5, diversity=0.0)
    naive_tracks = [TRACKS_BY_ID[r["track_id"]] for r in naive_results]
    naive_pairs = list(itertools.combinations(naive_tracks, 2))
    naive_avg_sim = sum(audio_similarity(a, b) for a, b in naive_pairs) / len(naive_pairs)

    assert mmr_avg_sim <= naive_avg_sim


def test_diversity_parameter_bounds_are_clamped():
    """diversity outside [0,1] should be clamped, not raise or produce
    nonsensical negative lambda values."""
    seed = TRACKS[0]["id"]
    results_low = recommend_next_track([seed], top_n=3, diversity=-5.0)
    results_high = recommend_next_track([seed], top_n=3, diversity=5.0)
    assert len(results_low) == 3
    assert len(results_high) == 3


def test_explain_reflects_full_history_not_just_last_seed():
    """If a candidate shares a tag ONLY with an older seed (not the
    last one), explain() must still mention it -- otherwise the
    explanation is out of sync with what multi-seed scoring actually
    used to compute the score."""
    tagged_tracks = [t for t in TRACKS if t["tags"]]
    older_seed = None
    candidate = None
    for a in tagged_tracks:
        for b in tagged_tracks:
            if a["id"] != b["id"] and (a["tags"] & b["tags"]):
                older_seed, candidate = a, b
                break
        if older_seed:
            break
    assert older_seed is not None, "test data must contain at least one shared-tag pair"

    disjoint_candidates = [
        t for t in TRACKS
        if t["id"] not in (older_seed["id"], candidate["id"])
        and not (t["tags"] & candidate["tags"])
    ]
    if not disjoint_candidates:
        pytest.skip("no disjoint-tag track available in current catalogue for this test")
    last_seed = disjoint_candidates[0]

    seed_ids = [older_seed["id"], last_seed["id"]]
    result = hybrid_score(seed_ids, candidate, "neutral", 0.5)
    result["id"] = candidate["id"]
    result["adjusted_score"] = result["score"]

    reason = explain(seed_ids, result, "neutral")
    shared = older_seed["tags"] & candidate["tags"]
    assert any(tag in reason for tag in shared) 