"""
weight_optimizer.py
--------------------
Turns NextTrack's fixed hybrid-score weights
(0.35 / 0.25 / 0.20 / 0.20, chosen by reasoned judgement in the Draft
Report, Section 3.4) into weights fit by cross-validated search against
real mined transition data, plus an ablation study that quantifies each
signal's individual contribution.

The draft's design chapter justified the fixed weights with literature
(Schedl et al. 2022; Whitman & Lawrence 2002) and reasoned argument, but
never tested whether those specific numbers were actually good numbers
-- only that a hybrid of these four signal *types* should outperform
any single one. This script closes that gap directly:

  1. Grid/coordinate-ascent search over the weight simplex (weights
     sum to 1, each in [0, 1]), evaluated with k-fold cross-validation
     on the real 1,951 mined transitions, optimizing Hit@1.
  2. An ablation study: re-run Hit@k with each signal's weight forced
     to 0 (and the rest renormalised), to show how much each signal
     individually contributes -- not just "the hybrid works" but
     "here is exactly how much each part is worth."
  3. Honest reporting of *both* outcomes: if the optimized weights
     beat the hand-picked ones only marginally, that is itself a
     valid, useful finding (it means the reasoned/literature-informed
     choice was already close to optimal) -- exactly the kind of
     result the module's own guidance says should be reported
     honestly rather than only favourable numbers being kept.

This also gives Evaluation something genuinely new: a controlled comparison between "expert-
chosen" and "data-optimized" weights, with train/test separation so
the number is not simply overfit to the evaluation set itself.

Run with:
    python weight_optimizer.py
"""

import random
import itertools
from collections import defaultdict

from data import TRACKS, TRACKS_BY_ID, TRANSITION_COUNTS
import scorer  # uses scorer.WEIGHT_AUDIO etc. and scorer.hybrid_score

random.seed(42)

DEFAULT_WEIGHTS = {
    "audio": scorer.WEIGHT_AUDIO,
    "tags": scorer.WEIGHT_TAGS,
    "sequence": scorer.WEIGHT_SEQUENCE,
    "context": scorer.WEIGHT_CONTEXT,
}

# Context is deliberately excluded from the *ranking* metric here: the
# offline Hit@k evaluation has no ground-truth mood/energy request
# attached to each real transition, so a weight search that included
# context would be optimizing against an undefined target for that
# term. Context's contribution is instead validated separately by
# mood_adherence_eval() in evaluate.py, unchanged from the draft. This
# scoping decision is stated explicitly here (and must be repeated in
# the report) rather than left implicit, since it is a real limitation
# of this method, not an oversight.
SEARCH_SIGNALS = ("audio", "tags", "sequence")
STEP = 0.05  # grid resolution


def _score_with_weights(seed_ids, candidate, weights):
    """Recomputes the *relatedness* part of hybrid_score using
    arbitrary weights, bypassing the module-level constants in
    scorer.py so many weight combinations can be tried without
    mutating global state."""
    audio, tags = scorer._weighted_audio_and_tags(seed_ids, candidate)
    sequence = scorer._weighted_sequence_score(seed_ids, candidate["id"])
    score = (
        weights["audio"] * audio
        + weights["tags"] * tags
        + weights["sequence"] * sequence
    )
    return score


def _real_next_tracks():
    next_map = defaultdict(set)
    for (a, b) in TRANSITION_COUNTS:
        next_map[a].add(b)
    return next_map


def _hit_at_k(seeds, weights, k=5):
    """Hit@k using arbitrary weights, restricted to SEARCH_SIGNALS
    (context excluded, see note above)."""
    next_map = _real_next_tracks()
    hits1 = hitsk = 0
    for seed in seeds:
        scored = []
        for t in TRACKS:
            if t["id"] == seed:
                continue
            s = _score_with_weights([seed], t, weights)
            scored.append((s, t["id"]))
        scored.sort(reverse=True)
        top_ids = [tid for _, tid in scored[:k]]
        real = next_map[seed]
        if top_ids and top_ids[0] in real:
            hits1 += 1
        if any(tid in real for tid in top_ids):
            hitsk += 1
    n = len(seeds)
    return hits1 / n, hitsk / n


def _weight_grid(step=STEP):
    """Yields every {audio, tags, sequence} combination on the simplex
    (summing to 1.0) at the given grid resolution. With step=0.05 this
    is C(20+2,2) = 231 combinations -- small enough to brute-force
    rather than needing a smarter optimizer, and exhaustive enough to
    not miss the optimum by more than the grid resolution."""
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            yield {
                "audio": round(i * step, 2),
                "tags": round(j * step, 2),
                "sequence": round(k * step, 2),
            }


def k_fold_search(k_folds=5, step=STEP, sample_size=None):
    """Cross-validated grid search: splits the seeds with a known real
    transition into k folds. For each fold, searches the weight grid
    using the OTHER folds (train), then evaluates the best-on-train
    weights on the held-out fold (test). Reports the average held-out
    Hit@1, which is a fair estimate of how the optimized weights would
    generalise -- not just how well they fit the exact data they were
    tuned on."""
    next_map = _real_next_tracks()
    seeds = [tid for tid in next_map if tid in TRACKS_BY_ID]
    random.shuffle(seeds)
    if sample_size:
        seeds = seeds[:sample_size]

    folds = [seeds[i::k_folds] for i in range(k_folds)]
    fold_results = []
    chosen_weights_per_fold = []

    for fold_idx in range(k_folds):
        test_seeds = folds[fold_idx]
        train_seeds = [s for i, f in enumerate(folds) if i != fold_idx for s in f]

        best_train_hit1 = -1.0
        best_weights = None
        for weights in _weight_grid(step):
            hit1, _ = _hit_at_k(train_seeds, weights, k=5)
            if hit1 > best_train_hit1:
                best_train_hit1 = hit1
                best_weights = weights

        test_hit1, test_hit5 = _hit_at_k(test_seeds, best_weights, k=5)
        fold_results.append((test_hit1, test_hit5))
        chosen_weights_per_fold.append(best_weights)

        print(f"Fold {fold_idx + 1}/{k_folds}: "
              f"best-on-train weights = {best_weights}, "
              f"held-out Hit@1 = {test_hit1:.1%}, Hit@5 = {test_hit5:.1%}")

    avg_hit1 = sum(h1 for h1, _ in fold_results) / k_folds
    avg_hit5 = sum(h5 for _, h5 in fold_results) / k_folds
    print(f"\nCross-validated average (optimized weights, held-out folds): "
          f"Hit@1 = {avg_hit1:.1%}, Hit@5 = {avg_hit5:.1%}")

    # Compare against the fixed, hand-chosen weights on the exact same
    # folds, so the comparison is apples-to-apples.
    baseline_results = [_hit_at_k(folds[i], DEFAULT_WEIGHTS, k=5) for i in range(k_folds)]
    base_hit1 = sum(h1 for h1, _ in baseline_results) / k_folds
    base_hit5 = sum(h5 for _, h5 in baseline_results) / k_folds
    print(f"Cross-validated average (draft's fixed weights {DEFAULT_WEIGHTS}): "
          f"Hit@1 = {base_hit1:.1%}, Hit@5 = {base_hit5:.1%}")

    delta = avg_hit1 - base_hit1
    print(f"\nDelta (optimized - fixed), Hit@1: {delta:+.1%}")
    if abs(delta) < 0.03:
        print("Interpretation: the literature-informed fixed weights were "
              "already close to optimal for this catalogue -- data-driven "
              "search does not meaningfully beat reasoned design here. "
              "This is reported as a genuine finding, not a failed "
              "experiment: it is evidence *for* the design decision made "
              "in Chapter 3, now backed by search rather than assumption.")
    else:
        print("Interpretation: data-driven search found weights that "
              "measurably outperform the reasoned fixed weights. Report "
              "this honestly in Chapter 5 as a limitation of the original "
              "design, and consider whether to adopt the optimized "
              "weights for the final submission (re-running the full "
              "48-test suite afterward, since several tests assert exact "
              "WEIGHT_* values).")

    return chosen_weights_per_fold, avg_hit1, base_hit1


def ablation_study(sample_size=100, k=5):
    """Zeroes out one signal at a time (renormalising the rest to sum
    to 1) and reports the Hit@1/Hit@5 drop, to quantify each signal's
    individual contribution -- direct evidence for the hybrid-signal
    design claim in Chapter 2/3, rather than an assertion."""
    next_map = _real_next_tracks()
    seeds = [tid for tid in next_map if tid in TRACKS_BY_ID]
    random.shuffle(seeds)
    seeds = seeds[:sample_size]

    print(f"\nAblation study (n={len(seeds)} seeds, k={k}):")
    full_hit1, full_hit5 = _hit_at_k(seeds, DEFAULT_WEIGHTS, k=k)
    print(f"  Full hybrid (audio={DEFAULT_WEIGHTS['audio']}, "
          f"tags={DEFAULT_WEIGHTS['tags']}, "
          f"sequence={DEFAULT_WEIGHTS['sequence']}): "
          f"Hit@1={full_hit1:.1%}, Hit@{k}={full_hit5:.1%}")

    for drop in SEARCH_SIGNALS:
        remaining = [s for s in SEARCH_SIGNALS if s != drop]
        total = sum(DEFAULT_WEIGHTS[s] for s in remaining)
        ablated_weights = {s: (DEFAULT_WEIGHTS[s] / total if s in remaining else 0.0)
                            for s in SEARCH_SIGNALS}
        hit1, hit5 = _hit_at_k(seeds, ablated_weights, k=k)
        print(f"  Without {drop:>8} (renormalised {ablated_weights}): "
              f"Hit@1={hit1:.1%} ({hit1 - full_hit1:+.1%}), "
              f"Hit@{k}={hit5:.1%} ({hit5 - full_hit5:+.1%})")

    # Single-signal-only versions, for the opposite comparison (does
    # any ONE signal alone match the hybrid?).
    print("\n  Single-signal-only comparisons:")
    for signal in SEARCH_SIGNALS:
        solo_weights = {s: (1.0 if s == signal else 0.0) for s in SEARCH_SIGNALS}
        hit1, hit5 = _hit_at_k(seeds, solo_weights, k=k)
        print(f"    {signal} only: Hit@1={hit1:.1%} ({hit1 - full_hit1:+.1%}), "
              f"Hit@{k}={hit5:.1%} ({hit5 - full_hit5:+.1%})")


if __name__ == "__main__":
    print("=" * 70)
    print("PART 1: Cross-validated weight optimization")
    print("=" * 70)
    k_fold_search(k_folds=5)

    print("\n" + "=" * 70)
    print("PART 2: Ablation study")
    print("=" * 70)
    ablation_study()
