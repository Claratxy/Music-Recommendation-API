"""
weight_optimizer_fixed.py
---------------------------
Fixes a data-leakage bug in the original weight_optimizer.py.

THE BUG (found by running the original against the real catalogue):
sequence-only weighting scored exactly 100.0% Hit@1/Hit@5 across every
cross-validation fold, with zero variance. That is not evidence the
sequence signal is powerful -- it is proof the evaluation was
circular. next_map (the "ground truth") is built directly from
TRANSITION_COUNTS, and sequence_score_single() also reads directly
from TRANSITION_COUNTS -- so "does the top pick match a known real
transition" is guaranteed to succeed whenever ranking is done by
sequence score, for any seed that has a recorded transition at all
(which is every seed used, by construction). The k-fold split held
out SEEDS but not the TRANSITION DATA itself, so the sequence signal
could always see the exact answer for every test seed.

THE FIX: split TRANSITION_COUNTS itself into train/test halves before
anything else. The sequence signal is computed using ONLY the train
half. Ground truth for evaluation is built ONLY from the test half.
A seed's own outgoing transitions are therefore genuinely unknown to
the scorer being tested -- this is now a fair test of whether each
signal predicts *held-out* real listening behaviour, not a test of
whether TRANSITION_COUNTS can predict itself.

This is worth keeping in the report regardless of what the corrected
numbers turn out to be: catching and fixing a circular evaluation is
itself a legitimate, citable methodological contribution for Chapter 5
("Evaluation methodology" or "Critical reflection"), and is far
stronger evidence of rigor than the original (invalid) 100% result
would have been.

Run with:
    python weight_optimizer_fixed.py
"""

import random
from collections import defaultdict

from data import TRACKS, TRACKS_BY_ID, TRANSITION_COUNTS
import scorer

random.seed(42)

DEFAULT_WEIGHTS = {
    "audio": scorer.WEIGHT_AUDIO,
    "tags": scorer.WEIGHT_TAGS,
    "sequence": scorer.WEIGHT_SEQUENCE,
}
SEARCH_SIGNALS = ("audio", "tags", "sequence")
STEP = 0.05


def _split_transitions(test_fraction=0.3, seed=7):
    """Splits the (seed_id, next_id) -> count dict into train/test
    halves by transition PAIR, not by seed -- a given seed can have
    some outgoing transitions in train and others in test, which is
    fine and realistic (a real user's history is never fully known
    either)."""
    rng = random.Random(seed)
    items = list(TRANSITION_COUNTS.items())
    rng.shuffle(items)
    split_point = int(len(items) * (1 - test_fraction))
    train = dict(items[:split_point])
    test = dict(items[split_point:])
    return train, test


def _sequence_score_from(transition_counts, last_track_id, candidate_id):
    """Same formula as scorer.sequence_score_single(), but reading
    from an arbitrary transition dict instead of the module-level
    TRANSITION_COUNTS -- so it can be pointed at the TRAIN split
    only."""
    count = transition_counts.get((last_track_id, candidate_id), 0)
    if count == 0:
        return 0.0
    outgoing = [c for (a, b), c in transition_counts.items() if a == last_track_id]
    max_count = max(outgoing) if outgoing else 1
    return count / max_count


def _score_with_weights(seed_ids, candidate, weights, train_transitions):
    audio, tags = scorer._weighted_audio_and_tags(seed_ids, candidate)
    sequence = _sequence_score_from(train_transitions, seed_ids[-1], candidate["id"])
    return (
        weights["audio"] * audio
        + weights["tags"] * tags
        + weights["sequence"] * sequence
    )


def _hit_at_k(seeds, weights, train_transitions, test_next_map, k=5):
    hits1 = hitsk = 0
    for seed in seeds:
        scored = []
        for t in TRACKS:
            if t["id"] == seed:
                continue
            s = _score_with_weights([seed], t, weights, train_transitions)
            scored.append((s, t["id"]))
        scored.sort(reverse=True)
        top_ids = [tid for _, tid in scored[:k]]
        real = test_next_map[seed]
        if top_ids and top_ids[0] in real:
            hits1 += 1
        if any(tid in real for tid in top_ids):
            hitsk += 1
    n = len(seeds)
    return hits1 / n, hitsk / n


def _weight_grid(step=STEP):
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            yield {
                "audio": round(i * step, 2),
                "tags": round(j * step, 2),
                "sequence": round(k * step, 2),
            }


def run(test_fraction=0.3, k=5):
    train_transitions, test_transitions = _split_transitions(test_fraction)

    test_next_map = defaultdict(set)
    for (a, b) in test_transitions:
        test_next_map[a].add(b)

    # Only evaluate on seeds whose TEST-half transition is to a track
    # that exists in the catalogue -- same eligibility rule as before,
    # just applied to the held-out half only.
    eligible_seeds = [s for s in test_next_map if s in TRACKS_BY_ID]
    print(f"Transition split: {len(train_transitions)} train, "
          f"{len(test_transitions)} test. "
          f"{len(eligible_seeds)} seeds have an eligible held-out transition.")

    if len(eligible_seeds) < 10:
        print("Too few eligible test seeds after the split -- lower "
              "test_fraction or accept that this catalogue's transition "
              "data is too sparse for a clean held-out sequence "
              "evaluation (worth reporting as a limitation either way).")
        return

    best_hit1 = -1.0
    best_weights = None
    for weights in _weight_grid():
        h1, _ = _hit_at_k(eligible_seeds, weights, train_transitions, test_next_map, k=k)
        if h1 > best_hit1:
            best_hit1 = h1
            best_weights = weights

    base_hit1, base_hit5 = _hit_at_k(eligible_seeds, DEFAULT_WEIGHTS, train_transitions, test_next_map, k=k)
    best_hit1_full, best_hit5_full = _hit_at_k(eligible_seeds, best_weights, train_transitions, test_next_map, k=k)

    print(f"\nFixed weights {DEFAULT_WEIGHTS}: Hit@1={base_hit1:.1%}, Hit@{k}={base_hit5:.1%}")
    print(f"Best weights found {best_weights}: Hit@1={best_hit1_full:.1%}, Hit@{k}={best_hit5_full:.1%}")
    print(f"Delta: {best_hit1_full - base_hit1:+.1%}")
    print("\n(Sanity check: if sequence-only weighting still scores ~100% "
          "here, the leak has not been fully closed -- check that no "
          "test-half transition pairs leaked into train_transitions.)")

    # Explicit sanity check against the leak this script was written to fix.
    sequence_only = {"audio": 0.0, "tags": 0.0, "sequence": 1.0}
    seq_hit1, _ = _hit_at_k(eligible_seeds, sequence_only, train_transitions, test_next_map, k=k)
    print(f"Sequence-only Hit@1 under this corrected split: {seq_hit1:.1%} "
          f"(should be well below 100% now -- compare to the original "
          f"script's 100.0% to see the size of the leak that was fixed)")

    return {
        "base_hit1": base_hit1,
        "best_hit1": best_hit1_full,
        "best_weights": best_weights,
        "delta": best_hit1_full - base_hit1,
        "sequence_only_hit1": seq_hit1,
        "n_test_seeds": len(eligible_seeds),
    }


if __name__ == "__main__":
    run()