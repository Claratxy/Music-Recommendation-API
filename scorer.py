"""
scorer.py 
--------------------
"""

import re
import math
import numpy as np
from data import TRACKS, TRACKS_BY_ID, TRANSITION_COUNTS

WEIGHT_AUDIO = 0.35
WEIGHT_TAGS = 0.25
WEIGHT_SEQUENCE = 0.20
WEIGHT_CONTEXT = 0.20
# These four weights sum to 1.0. Previously mood/energy (context) was
# only a +/-15% multiplier applied after the fact, which meant a
# request for "calm, low energy" barely changed the ranking versus the
# default. Making it a first-class weighted term (same footing as
# audio/tags/sequence) means mood actually competes for the top slot,
# and gives us something concrete to evaluate (see
# evaluation/evaluate.py::mood_adherence_eval).

# How quickly older seed tracks lose influence. 0.6 means each step
# further back in the history contributes 60% as much as the one
# after it (seed[-1]=1.0, seed[-2]=0.6, seed[-3]=0.36, ...).
RECENCY_DECAY = 0.6

# Cap on how many seeds contribute to sequence-score lookups, to keep
# this cheap even if track_ids grows beyond the current 5-item limit.
MAX_SEQUENCE_LOOKBACK = 5


def _seed_weights(seed_track_ids):
    """Returns [(track_id, weight), ...] most-recent-first, weights
    summing to 1.0, using exponential recency decay."""
    ordered = list(reversed(seed_track_ids))
    raw = [RECENCY_DECAY ** i for i in range(len(ordered))]
    total = sum(raw)
    return [(tid, w / total) for tid, w in zip(ordered, raw)]


def audio_similarity(track_a, track_b):
    tempo_diff = (track_a["tempo"] - track_b["tempo"]) / 200.0
    energy_diff = track_a["energy"] - track_b["energy"]
    valence_diff = track_a["valence"] - track_b["valence"]
    distance = math.sqrt(tempo_diff ** 2 + energy_diff ** 2 + valence_diff ** 2)
    max_distance = math.sqrt(1 ** 2 + 1 ** 2 + 1 ** 2)
    return 1 - min(distance / max_distance, 1.0)


def tag_overlap(track_a, track_b):
    tags_a, tags_b = track_a["tags"], track_b["tags"]
    if not tags_a or not tags_b:
        return 0.0
    intersection = tags_a & tags_b
    union = tags_a | tags_b
    return len(intersection) / len(union)


def sequence_score_single(last_track_id, candidate_id):
    """Original single-pair sequence score, kept for test compatibility."""
    count = TRANSITION_COUNTS.get((last_track_id, candidate_id), 0)
    if count == 0:
        return 0.0
    outgoing = [c for (a, b), c in TRANSITION_COUNTS.items() if a == last_track_id]
    max_count = max(outgoing) if outgoing else 1
    return count / max_count


# Backwards-compatible alias (existing tests import `sequence_score`).
sequence_score = sequence_score_single


def _weighted_sequence_score(seed_track_ids, candidate_id):
    """Sequence score across the whole history, most-recent seed
    weighted highest, so a strong transition two tracks back still
    counts for something instead of being invisible."""
    weighted = _seed_weights(seed_track_ids[-MAX_SEQUENCE_LOOKBACK:])
    total = 0.0
    for tid, weight in weighted:
        total += weight * sequence_score_single(tid, candidate_id)
    return min(total, 1.0)


def context_adjustment(track, mood, energy_target):
    energy_match = 1 - abs(track["energy"] - energy_target)
    mood_to_valence = {
        "energetic": 0.75,
        "happy": 0.8,
        "calm": 0.35,
        "melancholy": 0.3,
        "neutral": 0.5,
    }
    target_valence = mood_to_valence.get(mood, 0.5)
    mood_match = 1 - abs(track["valence"] - target_valence)
    return (energy_match + mood_match) / 2


def _weighted_audio_and_tags(seed_track_ids, candidate):
    """Audio similarity and tag overlap averaged across the seed
    sequence, weighted by recency, instead of only the last track."""
    weighted = _seed_weights(seed_track_ids)
    audio_total = 0.0
    tags_total = 0.0
    for tid, weight in weighted:
        seed = TRACKS_BY_ID[tid]
        audio_total += weight * audio_similarity(seed, candidate)
        tags_total += weight * tag_overlap(seed, candidate)
    return audio_total, tags_total


def hybrid_score(seed_track_ids, candidate, mood, energy_target):
    """
    Combines audio similarity, tag overlap, sequence score, and mood/
    energy context against the WHOLE seed history (recency-weighted)
    into one hybrid score between 0 and 1. Context (mood/energy) is
    now a first-class weighted term rather than a post-hoc multiplier,
    so a strong mood/energy request can genuinely change the winner,
    not just fine-tune it by a few percent.

    `relatedness` deliberately excludes context -- it represents how
    related the candidate is to the listening history itself, which is
    what novelty_rerank() uses to decide whether a track is even a
    plausible candidate for boosting. Mood/energy is a per-request
    preference, not part of "relatedness to the history", so mixing
    it in would let two people with identical listening history but
    different mood sliders get different novelty floors for no good
    reason.
    """
    audio, tags = _weighted_audio_and_tags(seed_track_ids, candidate)
    sequence = _weighted_sequence_score(seed_track_ids, candidate["id"])
    context = context_adjustment(candidate, mood, energy_target)

    relatedness = (
        WEIGHT_AUDIO * audio
        + WEIGHT_TAGS * tags
        + WEIGHT_SEQUENCE * sequence
    )
    final_score = relatedness + WEIGHT_CONTEXT * context
    final_score = max(0.0, min(1.0, final_score))

    return {
        "score": final_score,
        "audio": audio,
        "tags": tags,
        "sequence": sequence,
        "context": context,
        "relatedness": relatedness,
    }


def novelty_rerank(scored_candidates, novelty):
    if not scored_candidates:
        return []
 
    relatedness_values = [c["relatedness"] for c in scored_candidates]
    max_relatedness = max(relatedness_values) if relatedness_values else 0.0
    relatedness_floor = 0.5 * max_relatedness
 
    pops = [_pop(item) for item in scored_candidates]
    min_pop, max_pop = min(pops), max(pops)
    pop_range = max(max_pop - min_pop, 1e-6)
 
    adjusted = []
    for item in scored_candidates:
        popularity = _pop(item)
        relatedness = item["relatedness"]
        if relatedness < relatedness_floor or max_relatedness == 0:
            novelty_boost = 0.0
        else:
            relative_rarity = (max_pop - popularity) / pop_range
            # squared relatedness ratio: the one-line fix for the
            # convergence problem found in Section 5.4 of the draft.
            relatedness_ratio = (relatedness / max_relatedness) ** 2
            novelty_boost = novelty * relative_rarity * relatedness_ratio * item["score"]
        adjusted_score = (1 - novelty) * item["score"] + novelty_boost
        adjusted.append({**item, "adjusted_score": adjusted_score})
 
    adjusted.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return adjusted

def _pop(item):
    # kept as a tiny helper so this file doesn't need to import
    # TRACKS_BY_ID -- in scorer.py itself, just keep using
    # TRACKS_BY_ID[item["id"]]["popularity"] as before.
    from data import TRACKS_BY_ID
    return TRACKS_BY_ID[item["id"]]["popularity"]

def build_feature_matrix(tracks):
    """Pre-computes a (N, 3) numpy array of [tempo/200, energy, valence]
    for every track in a fixed order, plus an id->row index map. Built
    once at import time in data.py and reused by every request, instead
    of being rebuilt per-request."""
    ids = [t["id"] for t in tracks]
    matrix = np.array(
        [[t["tempo"] / 200.0, t["energy"], t["valence"]] for t in tracks],
        dtype=np.float64,
    )
    index = {tid: i for i, tid in enumerate(ids)}
    return ids, matrix, index
 
 
_MAX_DIST = np.sqrt(3.0)

def audio_similarity_batch(seed_vec, all_vecs):
    """Vectorized version of audio_similarity(): compares one seed
    feature vector against every row of all_vecs at once.
 
    Returns an array of similarities in [0, 1], same formula as the
    scalar audio_similarity() (verified equal in
    tests/test_scorer_vectorized.py), but computed in one numpy call
    instead of a Python loop over every candidate. For a 174-track
    catalogue this is not strictly necessary for interactive-speed use,
    but it is what makes weight_optimizer.py's grid search (which
    re-scores the whole catalogue for every seed, for every weight
    combination, across a k-fold split -- tens of thousands of scoring
    calls) run in seconds rather than minutes, and is the kind of
    micro-optimization a production-grade recommender needs once the
    catalogue or the weight-search space grows.
    """
    diff = seed_vec - all_vecs
    dist = np.linalg.norm(diff, axis=1)
    return 1.0 - np.minimum(dist / _MAX_DIST, 1.0)

def _mmr_select(ranked_candidates, top_n, lambda_relevance=0.7):
    """
    Real Maximal Marginal Relevance selection (Carbonell & Goldstein,
    1998): repeatedly picks the candidate that maximises

        lambda * relevance(c) - (1 - lambda) * max_sim(c, selected)

    so each additional pick trades off staying relevant against being
    audio-similar to tracks already chosen -- this is what
    /recommend/queue's docstring has always claimed, now actually
    implemented.
    """
    if not ranked_candidates:
        return []

    pool = list(ranked_candidates)
    selected = [pool.pop(0)]  # top-ranked track always taken first

    while pool and len(selected) < top_n:
        best_item = None
        best_mmr = float("-inf")
        for item in pool:
            candidate_track = TRACKS_BY_ID[item["id"]]
            max_sim = max(
                audio_similarity(candidate_track, TRACKS_BY_ID[s["id"]])
                for s in selected
            )
            mmr = lambda_relevance * item["adjusted_score"] - (1 - lambda_relevance) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_item = item
        selected.append(best_item)
        pool.remove(best_item)

    return selected


def explain(seed_track_ids, top_result, mood):
    candidate = TRACKS_BY_ID[top_result["id"]]

    # Aggregate shared tags across the WHOLE weighted history, not just
    # the last seed, so the explanation matches what actually drove the
    # score under multi-seed scoring. Tags are collected with the same
    # recency weighting as the score itself, so a tag shared with an
    # older seed is mentioned but a tag shared with the most recent
    # seed is listed first.
    weighted = _seed_weights(seed_track_ids)
    shared_tags_by_recency = []
    seen_tags = set()
    for tid, _weight in weighted:
        seed = TRACKS_BY_ID[tid]
        for tag in sorted(seed["tags"] & candidate["tags"]):
            if tag not in seen_tags:
                shared_tags_by_recency.append(tag)
                seen_tags.add(tag)

    reasons = []
    if top_result["audio"] > 0.7:
        reasons.append("similar tempo and energy to the recent tracks")
    if shared_tags_by_recency:
        reasons.append(f"shared {', '.join(shared_tags_by_recency)} tag(s)")
    if top_result["sequence"] > 0:
        reasons.append("often follows tracks like these in other playlists")
    if top_result["context"] > 0.8:
        article = "an" if mood[:1].lower() in "aeiou" else "a"
        reasons.append(f"strong match for {article} {mood} mood at the requested energy")
    if not reasons:
        article = "an" if mood[:1].lower() in "aeiou" else "a"
        reasons.append(f"a good match for {article} {mood} mood right now")

    novelty_note = ""
    if top_result["adjusted_score"] > top_result["score"] + 0.05:
        novelty_note = "; boosted for variety, as requested"
    elif top_result["adjusted_score"] < top_result["score"] - 0.05:
        novelty_note = "; a strong familiar-style match despite the novelty request"

    return "; ".join(reasons) + novelty_note


def _normalize_title(title):
    return re.sub(r"\s*[-(].*$", "", title).strip().lower()


def recommend_next_track(seed_track_ids, mood="neutral", energy=0.5, novelty=0.3, top_n=1, diversity=0.3):
    seed_titles = {
        _normalize_title(TRACKS_BY_ID[sid]["title"]) + "::" + TRACKS_BY_ID[sid]["artist"].lower()
        for sid in seed_track_ids
    }
    candidate_ids = [
        t["id"] for t in TRACKS
        if t["id"] not in seed_track_ids
        and (_normalize_title(t["title"]) + "::" + t["artist"].lower()) not in seed_titles
    ]

    scored = []
    for cid in candidate_ids:
        candidate = TRACKS_BY_ID[cid]
        result = hybrid_score(seed_track_ids, candidate, mood, energy)
        scored.append({"id": cid, **result})

    reranked = novelty_rerank(scored, novelty)

    if top_n == 1:
        top_results = reranked[:1]
    else:
        # Diversify beyond the top pick using MMR, instead of just
        # taking the next N highest-scored (often near-duplicates).
        # diversity=0.0 -> lambda_relevance=1.0 (pure relevance, same as
        #   a plain top-N slice).
        # diversity=1.0 -> lambda_relevance=0.0 (pure diversity, ignores
        #   relevance after the first pick).
        # Default 0.3 keeps most weight on relevance while still
        # spreading out the remaining slots.
        diversity = max(0.0, min(1.0, diversity))
        top_results = _mmr_select(reranked, top_n, lambda_relevance=1.0 - diversity)

    output = []
    for result in top_results:
        track = TRACKS_BY_ID[result["id"]]
        output.append({
            "track_id": result["id"],
            "spotify_id": track.get("spotify_id"),
            "title": track["title"],
            "artist": track["artist"],
            "confidence_score": round(result["adjusted_score"], 2),
            "reason": explain(seed_track_ids, result, mood),
            "breakdown": {
                "audio": round(result["audio"], 2),
                "tags": round(result["tags"], 2),
                "sequence": round(result["sequence"], 2),
                "context": round(result["context"], 2),
            },
            "tags": sorted(track["tags"]),
            "tag_source": track.get("tag_source", "unknown"),
        })
    return output


# ---------------------------------------------------------------------
# NOTE FOR evaluate.py: the two functions below are not part of the
# scorer itself -- copy them into evaluation/evaluate.py (see the
# accompanying guidance) so the evaluation actually exercises
# multi-seed weighting and MMR diversification, neither of which
# hit_rate_eval()/novelty_sensitivity_eval() currently touch (they
# only ever call recommend_next_track with a single-track seed list
# and top_n=1).
# ---------------------------------------------------------------------