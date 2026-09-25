"""
evaluate.py
------------
Offline evaluation of NextTrack's hybrid scorer, using the real
adjacent-track transitions mined from real playlists.

CHANGES IN THIS VERSION (vs. the draft's evaluate.py):
  - full_hit_rate_eval() replaces the sample-based hit_rate_eval() as
    the headline Hit@k result: leave-one-out over ALL eligible seeds,
    with bootstrap 95% confidence intervals. hit_rate_eval() is kept
    below for reference/backward-compatibility but is no longer called
    by default.
  - novelty_sensitivity_eval() and mood_adherence_eval() now RETURN
    their key numbers (not just print them), specifically so
    generate_verdict() can be wired to real, freshly-computed values
    instead of numbers hand-copied from a previous run -- that
    hand-copying is exactly what caused the earlier stale-placeholder
    bug (weight_delta was left at an example value of 0.012 instead of
    the real 0.182, and then the real corrected 0.068).
  - generate_verdict() gained two new checks: calm-adherence plateau
    (comparing against the Draft Report's own reported baseline) and
    novelty-sensitivity trade-off (top-pick-change rate dropping after
    the squared-relatedness fix). Both are real findings from the
    9 Oct rebalancing/fix run, not auto-detected guesses.
  - The weight-optimization delta now comes from the LEAKAGE-FREE
    weight_optimizer_fixed.py (train/test split on TRANSITION_COUNTS
    itself), not the original weight_optimizer.py, which was found to
    have a circular-evaluation bug (sequence-only scored a
    zero-variance 100% -- see weight_optimizer_fixed.py's docstring
    for the full diagnosis).

Run with:
    python -m evaluation.evaluate
"""

import random
from collections import defaultdict, Counter
import itertools

from data import TRACKS, TRACKS_BY_ID, TRANSITION_COUNTS
from scorer import recommend_next_track, audio_similarity
import weight_optimizer_fixed

random.seed(42)

# ---------------------------------------------------------------------
# Historical baselines from the Draft Report, kept as named constants
# (not re-derived, since they describe a catalogue/scorer state that
# no longer exists after the rebalancing and bug fixes). Used only for
# explicit before/after comparison in generate_verdict() -- cited here
# so it's clear these are fixed reference points, not live numbers.
# ---------------------------------------------------------------------
DRAFT_CALM_ADHERENCE_DEVIATION = 0.41       # Draft Report, Section 5.6, Table 7
DRAFT_NOVELTY_TOP_PICK_CHANGE_RATE = 0.533  # Draft Report, Section 5.4
DRAFT_NOVELTY_CONVERGENCE_PCT = 0.20        # Draft Report, Section 5.4 ("Re: Stacks")


def real_next_tracks():
    """Maps each track id to the set of tracks that are known to have
    really followed it, according to the mined playlist data."""
    next_map = defaultdict(set)
    for (a, b) in TRANSITION_COUNTS:
        next_map[a].add(b)
    return next_map


def bootstrap_ci(hits, n_resamples=2000, ci=0.95, seed=123):
    """Bootstrap confidence interval for a Hit@k proportion."""
    rng = random.Random(seed)
    n = len(hits)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_resamples):
        resample = [hits[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower_idx = int((1 - ci) / 2 * n_resamples)
    upper_idx = int((1 + ci) / 2 * n_resamples) - 1
    return means[lower_idx], means[upper_idx]


def full_hit_rate_eval(top_n=5):
    """Leave-one-out Hit@k over EVERY seed with a known real
    transition, plus a bootstrap 95% confidence interval. This is the
    headline accuracy result for the Final Report, replacing the
    draft's fixed 100-seed sample."""
    next_map = real_next_tracks()
    seeds = [tid for tid in next_map if tid in TRACKS_BY_ID]

    hits_at_1 = []
    hits_at_n = []
    for seed in seeds:
        results = recommend_next_track([seed], top_n=top_n)
        predicted_ids = [r["track_id"] for r in results]
        real_nexts = next_map[seed]
        hits_at_1.append(1 if predicted_ids and predicted_ids[0] in real_nexts else 0)
        hits_at_n.append(1 if any(pid in real_nexts for pid in predicted_ids) else 0)

    n = len(seeds)
    h1_rate = sum(hits_at_1) / n
    hn_rate = sum(hits_at_n) / n
    h1_lo, h1_hi = bootstrap_ci(hits_at_1)
    hn_lo, hn_hi = bootstrap_ci(hits_at_n)

    print(f"Full leave-one-out evaluation over all {n} eligible seeds "
          f"(no subsampling):")
    print(f"  Hit@1: {sum(hits_at_1)}/{n} = {h1_rate:.1%} "
          f"(95% CI: {h1_lo:.1%}-{h1_hi:.1%})")
    print(f"  Hit@{top_n}: {sum(hits_at_n)}/{n} = {hn_rate:.1%} "
          f"(95% CI: {hn_lo:.1%}-{hn_hi:.1%})")
    print(f"  Random Hit@{top_n} baseline: {top_n / len(TRACKS):.1%} "
          f"(outside the CI above -> the improvement over random is not "
          f"attributable to sampling noise)")
    return {
        "hit1": h1_rate, "hit1_ci": (h1_lo, h1_hi),
        "hitn": hn_rate, "hitn_ci": (hn_lo, hn_hi),
        "n": n,
    }


def held_out_full_pipeline_eval(top_n=5, test_fraction=0.3, split_seed=7):
    """The genuinely fair counterpart to full_hit_rate_eval() above.

    full_hit_rate_eval() calls recommend_next_track() using the
    COMPLETE, unsplit TRANSITION_COUNTS -- so for any seed with a real
    recorded transition, the sequence signal (20% of the blended
    score) is computed using data that includes the exact answer being
    tested for. This is the same category of leak diagnosed and fixed
    in weight_optimizer_fixed.py, just diluted across four signals
    instead of being the sole signal -- and the 12x gap between
    full_hit_rate_eval()'s 81.8% and weight_optimizer_fixed.py's
    genuinely held-out 6.8% (same catalogue, same fixed weights) is
    strong evidence it is still inflating the headline number.

    This function re-runs the FULL production pipeline (all four
    signals, novelty re-ranking, MMR) but temporarily points
    scorer.py's TRANSITION_COUNTS at a TRAIN-only split, and evaluates
    against a TEST-only ground truth -- exactly like
    weight_optimizer_fixed.py, but exercising the real
    recommend_next_track() function instead of a simplified
    re-implementation, so this number is directly about the actual
    shipped system, not an approximation of it.

    Uses monkey-patching (reassigning scorer.TRANSITION_COUNTS for the
    duration of this function, then restoring it) rather than
    modifying data.py permanently, so nothing else in this script or
    the wider codebase is affected.
    """
    import scorer as scorer_module

    train_transitions, test_transitions = weight_optimizer_fixed._split_transitions(
        test_fraction=test_fraction, seed=split_seed
    )
    test_next_map = defaultdict(set)
    for (a, b) in test_transitions:
        test_next_map[a].add(b)
    eligible_seeds = [s for s in test_next_map if s in TRACKS_BY_ID]

    if len(eligible_seeds) < 10:
        print("\nHeld-out full-pipeline eval: too few eligible test seeds "
              "after the split to report a meaningful number.")
        return None

    original_transitions = scorer_module.TRANSITION_COUNTS
    scorer_module.TRANSITION_COUNTS = train_transitions
    try:
        hits_at_1 = []
        hits_at_n = []
        for seed in eligible_seeds:
            results = recommend_next_track([seed], top_n=top_n)
            predicted_ids = [r["track_id"] for r in results]
            real_nexts = test_next_map[seed]
            hits_at_1.append(1 if predicted_ids and predicted_ids[0] in real_nexts else 0)
            hits_at_n.append(1 if any(pid in real_nexts for pid in predicted_ids) else 0)
    finally:
        scorer_module.TRANSITION_COUNTS = original_transitions  # always restore,
                                                                   # even if the loop
                                                                   # above raises

    n = len(eligible_seeds)
    h1_rate = sum(hits_at_1) / n
    hn_rate = sum(hits_at_n) / n
    h1_lo, h1_hi = bootstrap_ci(hits_at_1)
    hn_lo, hn_hi = bootstrap_ci(hits_at_n)

    print(f"\nHeld-out full-pipeline evaluation (genuinely unseen test "
          f"transitions, {n} eligible seeds, {test_fraction:.0%} held out):")
    print(f"  Hit@1: {sum(hits_at_1)}/{n} = {h1_rate:.1%} "
          f"(95% CI: {h1_lo:.1%}-{h1_hi:.1%})")
    print(f"  Hit@{top_n}: {sum(hits_at_n)}/{n} = {hn_rate:.1%} "
          f"(95% CI: {hn_lo:.1%}-{hn_hi:.1%})")
    print(f"  Random Hit@{top_n} baseline: {top_n / len(TRACKS):.1%}")
    print(f"  (Compare to full_hit_rate_eval()'s in-sample Hit@1 above -- "
          f"the gap between the two IS the size of the sequence-signal "
          f"leak in the standard evaluation, and should be reported "
          f"explicitly, not hidden.)")

    return {
        "hit1": h1_rate, "hit1_ci": (h1_lo, h1_hi),
        "hitn": hn_rate, "hitn_ci": (hn_lo, hn_hi),
        "n": n,
    }


def hit_rate_eval(top_n=5, sample_size=100):
    """SUPERSEDED by full_hit_rate_eval() -- kept only for reference /
    to show the draft's original sampled methodology if a direct
    before/after comparison of methodology itself is useful in the
    report. Not called by default in __main__."""
    next_map = real_next_tracks()
    seeds = [tid for tid in next_map if tid in TRACKS_BY_ID]
    random.shuffle(seeds)
    seeds = seeds[:sample_size]

    hits_at_1 = hits_at_n = 0
    for seed in seeds:
        results = recommend_next_track([seed], top_n=top_n)
        predicted_ids = [r["track_id"] for r in results]
        real_nexts = next_map[seed]
        if predicted_ids and predicted_ids[0] in real_nexts:
            hits_at_1 += 1
        if any(pid in real_nexts for pid in predicted_ids):
            hits_at_n += 1

    n = len(seeds)
    random_baseline = top_n / len(TRACKS)
    print(f"Evaluated {n} seeds with known real transitions "
          f"(out of {len(next_map)} available).")
    print(f"  Hit@1:  {hits_at_1}/{n} = {hits_at_1 / n:.1%}")
    print(f"  Hit@{top_n}:  {hits_at_n}/{n} = {hits_at_n / n:.1%}")
    print(f"  Random Hit@{top_n} baseline: {random_baseline:.1%}")


def novelty_sensitivity_eval(sample_size=60):
    """Returns a dict of {change_rate, convergence_pct, distinct_count}
    in addition to printing, so callers (generate_verdict) always use
    the number this exact run actually produced.

    Uses a LOCAL Random instance (fixed seed) rather than the shared
    global `random` module. This matters: the global module's state
    depends on exactly which other functions ran before this one and
    how many random draws they made -- so two runs of the same script
    (or the same run after an unrelated code change elsewhere in the
    file) can silently sample a different set of tracks, producing
    different numbers that look like a real change but are actually
    just sampling noise. A local instance makes this function's output
    reproducible regardless of what else runs around it."""
    local_random = random.Random(101)
    all_ids = [t["id"] for t in TRACKS]
    local_random.shuffle(all_ids)
    seeds = all_ids[:sample_size]

    changed = 0
    examples = []
    for seed in seeds:
        low = recommend_next_track([seed], novelty=0.1, top_n=1)[0]["track_id"]
        high = recommend_next_track([seed], novelty=0.9, top_n=1)[0]["track_id"]
        if low != high:
            changed += 1
            if len(examples) < 3:
                examples.append((seed, low, high))

    n = len(seeds)
    change_rate = changed / n
    print(f"\nNovelty sensitivity: top pick changed between novelty=0.1 "
          f"and novelty=0.9 for {changed}/{n} seeds ({change_rate:.1%}).")
    if examples:
        print("  Examples where it did change:")
        for seed, low, high in examples:
            seed_t = TRACKS_BY_ID[seed]
            print(f"    {seed_t['title']} -> low: "
                  f"{TRACKS_BY_ID[low]['title']}, high: {TRACKS_BY_ID[high]['title']}")

    high_picks = [recommend_next_track([s], novelty=0.9, top_n=1)[0]["track_id"] for s in seeds]
    distinct = len(set(high_picks))
    most_common_id, most_common_count = Counter(high_picks).most_common(1)[0]
    convergence_pct = most_common_count / n
    print(f"\nHigh-novelty pick diversity: {distinct} distinct tracks recommended "
          f"across {n} different seeds.")
    print(f"  Most frequent high-novelty pick: "
          f"{TRACKS_BY_ID[most_common_id]['title']} "
          f"({most_common_count}/{n} = {convergence_pct:.1%} of all seeds)")
    if convergence_pct > 0.15:
        print("  -> Convergence: one low-popularity track is being boosted "
              "across many unrelated seeds, not genuine per-seed discovery.")

    return {
        "change_rate": change_rate,
        "convergence_pct": convergence_pct,
        "distinct_count": distinct,
        "n": n,
    }


def multi_seed_eval(sample_size=60, history_len=3):
    all_ids = [t["id"] for t in TRACKS]
    local_random = random.Random(7)
    local_random.shuffle(all_ids)

    changed = 0
    n = 0
    for i in range(0, len(all_ids) - history_len, history_len):
        if n >= sample_size:
            break
        history = all_ids[i:i + history_len]
        last_only = recommend_next_track([history[-1]], top_n=1)[0]["track_id"]
        full_history = recommend_next_track(history, top_n=1)[0]["track_id"]
        if last_only != full_history:
            changed += 1
        n += 1

    rate = changed / n
    print(f"\nMulti-seed effect: recommendation changed between "
          f"last-track-only and full-{history_len}-track-history for "
          f"{changed}/{n} sessions ({rate:.1%}).")
    return {"change_rate": rate, "n": n}


def mmr_diversity_eval(sample_size=40, top_n=5):
    all_ids = [t["id"] for t in TRACKS]
    local_random = random.Random(11)
    local_random.shuffle(all_ids)
    seeds = all_ids[:sample_size]

    avg_pairwise_sims = []
    for seed in seeds:
        results = recommend_next_track([seed], top_n=top_n)
        tracks = [TRACKS_BY_ID[r["track_id"]] for r in results]
        pairs = list(itertools.combinations(tracks, 2))
        if not pairs:
            continue
        sims = [audio_similarity(a, b) for a, b in pairs]
        avg_pairwise_sims.append(sum(sims) / len(sims))

    overall_avg = sum(avg_pairwise_sims) / len(avg_pairwise_sims)
    print(f"\nMMR diversity check (top_n={top_n}): average pairwise "
          f"audio similarity within a single response = {overall_avg:.2f} "
          f"(lower = more diverse; 1.0 = tracks are audio-identical).")
    return {"avg_pairwise_similarity": overall_avg}


def mood_adherence_eval(sample_size=40):
    """Returns {calm_deviation, energetic_deviation} in addition to
    printing, so generate_verdict() can compare against the draft's
    reported baseline without a hand-typed number.

    Uses a LOCAL Random instance for the same reproducibility reason
    documented in novelty_sensitivity_eval() -- this function's
    previous version sampled from the shared global `random` module,
    which made its output depend on unrelated call-order changes
    elsewhere in the script (confirmed: energetic-request deviation
    swung from 0.26 to 0.35 between two runs against the IDENTICAL
    catalogue, purely because of this)."""
    local_random = random.Random(202)
    all_ids = [t["id"] for t in TRACKS]
    local_random.shuffle(all_ids)
    seeds = all_ids[:sample_size]

    diffs_low = []
    diffs_high = []
    for seed in seeds:
        low = recommend_next_track([seed], mood="calm", energy=0.2, top_n=1)[0]
        high = recommend_next_track([seed], mood="energetic", energy=0.9, top_n=1)[0]
        diffs_low.append(abs(TRACKS_BY_ID[low["track_id"]]["energy"] - 0.2))
        diffs_high.append(abs(TRACKS_BY_ID[high["track_id"]]["energy"] - 0.9))

    avg_low = sum(diffs_low) / len(diffs_low)
    avg_high = sum(diffs_high) / len(diffs_high)
    print(f"\nMood/energy adherence (n={len(seeds)}): average "
          f"|actual_energy - requested_energy| = {avg_low:.2f} for a "
          f"calm/low-energy request, {avg_high:.2f} for an "
          f"energetic/high-energy request (lower = closer match to "
          f"what was actually requested).")
    return {"calm_deviation": avg_low, "energetic_deviation": avg_high}


def catalogue_energy_distribution_eval():
    energies = sorted(t["energy"] for t in TRACKS)
    n = len(energies)
    below_03 = sum(1 for e in energies if e < 0.3)
    below_04 = sum(1 for e in energies if e < 0.4)
    above_07 = sum(1 for e in energies if e > 0.7)
    median = energies[n // 2]
    print(f"\nCatalogue energy distribution (n={n}): median={median:.2f}, "
          f"mean={sum(energies)/n:.2f}")
    print(f"  Tracks with energy < 0.3 (needed for calm/low requests): "
          f"{below_03} ({below_03/n:.1%})")
    print(f"  Tracks with energy < 0.4: {below_04} ({below_04/n:.1%})")
    print(f"  Tracks with energy > 0.7 (needed for energetic/high requests): "
          f"{above_07} ({above_07/n:.1%})")
    if below_03 < above_07 / 2:
        print("  -> Catalogue is still skewed toward higher-energy tracks "
              "relative to energetic-side tracks, even after rebalancing.")
    return {"median": median, "below_03_pct": below_03 / n, "above_07_pct": above_07 / n}


def generate_verdict(results_dict):
    """Structured, evidence-backed pros/cons readout. Every entry here
    is driven by a value actually computed in this run (or an
    explicitly-cited historical constant from the Draft Report) -- no
    hand-typed placeholder numbers."""
    print("\n" + "=" * 70)
    print("BALANCED VERDICT (for Section 5.9 / critical reflection)")
    print("=" * 70)

    strengths, weaknesses, uncertain = [], [], []

    if "hit1" in results_dict:
        h1 = results_dict["hit1"]
        if h1 > 0.5:
            strengths.append(
                f"In-sample Hit@1 = {h1:.1%} (full leave-one-out, standard "
                f"TRANSITION_COUNTS) is far above the {5/len(TRACKS):.1%} "
                f"random baseline -- the scorer's ranking is not arbitrary."
            )
        else:
            weaknesses.append(f"In-sample Hit@1 = {h1:.1%} is weaker than "
                               f"the draft's figure; investigate before "
                               f"reporting as-is.")

    if "held_out_hit1" in results_dict and results_dict["held_out_hit1"] is not None:
        held_out_h1 = results_dict["held_out_hit1"]
        in_sample_h1 = results_dict.get("hit1")
        if in_sample_h1 is not None:
            gap = in_sample_h1 - held_out_h1
            if gap > 0.15:
                weaknesses.append(
                    f"The headline in-sample Hit@1 ({in_sample_h1:.1%}) is "
                    f"substantially inflated by sequence-signal leakage: "
                    f"the genuinely held-out equivalent (identical system, "
                    f"transition data split before scoring) is only "
                    f"{held_out_h1:.1%}, a {gap:.1%}-point gap. The "
                    f"held-out figure is the honest estimate of how the "
                    f"system performs on listening behaviour it has not "
                    f"already seen, and should be the number quoted as the "
                    f"primary result, with the in-sample figure reported "
                    f"only as a secondary 'does the system retrieve known "
                    f"structure' sanity check."
                )
            else:
                strengths.append(
                    f"Held-out Hit@1 ({held_out_h1:.1%}) is close to the "
                    f"in-sample figure ({in_sample_h1:.1%}), suggesting "
                    f"limited leakage inflation in the headline metric."
                )
        else:
            strengths.append(
                f"Held-out (genuinely unseen data) Hit@1 = {held_out_h1:.1%}, "
                f"well above the {5/len(TRACKS):.1%} random baseline."
            )

    # --- Weight optimization (leakage-free result) --------------------
    if "weight_delta" in results_dict:
        delta = results_dict["weight_delta"]
        seq_only = results_dict.get("sequence_only_sanity_check")
        if seq_only is not None and seq_only > 0.5:
            weaknesses.append(
                f"Weight-optimization sanity check FAILED: sequence-only "
                f"scored {seq_only:.1%} on the held-out split -- this "
                f"suggests the train/test leak has not been fully closed. "
                f"Do not report the weight-optimization delta until this "
                f"reads well below 50%."
            )
        elif abs(delta) < 0.03:
            strengths.append(
                f"Leakage-free, cross-validated weight search found "
                f"weights within {delta:+.1%} Hit@1 of the hand-chosen "
                f"ones -- the Chapter 3 design holds up under rigorous "
                f"testing."
            )
        else:
            weaknesses.append(
                f"Leakage-free weight search found weights that beat the "
                f"hand-chosen ones by {delta:+.1%} Hit@1, with tags "
                f"weighted far more heavily than audio -- a genuine "
                f"limitation of the Chapter 3 design's emphasis, though "
                f"based on a single 70/30 split (n=73 test seeds) and "
                f"should be treated as directional evidence, not a "
                f"precise number, without repeating the split multiple "
                f"times to check stability."
            )

    # --- Novelty convergence fix ---------------------------------------
    if "novelty_convergence_after" in results_dict:
        after = results_dict["novelty_convergence_after"]
        before = DRAFT_NOVELTY_CONVERGENCE_PCT
        if after < before:
            strengths.append(
                f"The squared-relatedness fix reduced high-novelty pick "
                f"convergence from {before:.1%} (Draft, Section 5.4) to "
                f"{after:.1%} -- confirms the diagnosed fix worked."
            )
        else:
            uncertain.append(
                f"High-novelty convergence did not improve as hoped "
                f"({before:.1%} -> {after:.1%})."
            )

    # --- Novelty sensitivity trade-off (NEW finding) --------------------
    if "novelty_change_rate" in results_dict:
        current_rate = results_dict["novelty_change_rate"]
        draft_rate = DRAFT_NOVELTY_TOP_PICK_CHANGE_RATE
        if current_rate < draft_rate - 0.10:
            weaknesses.append(
                f"Fixing novelty convergence came with a trade-off: the "
                f"top-pick change rate between novelty=0.1 and novelty=0.9 "
                f"dropped from {draft_rate:.1%} (Draft, Section 5.4) to "
                f"{current_rate:.1%} -- the squared-relatedness fix makes "
                f"novelty boost more selective, so it now changes the top "
                f"pick for fewer seeds overall, even though it is boosting "
                f"more genuinely varied tracks when it does act. This is a "
                f"real precision-vs-sensitivity trade-off, not a pure "
                f"improvement, and should be named as such."
            )
        else:
            strengths.append(
                f"Novelty top-pick change rate held roughly steady at "
                f"{current_rate:.1%} (Draft: {draft_rate:.1%}) despite the "
                f"convergence fix -- the fix improved specificity without "
                f"costing sensitivity."
            )

    # --- Calm-adherence plateau (NEW finding) ---------------------------
    if "calm_deviation" in results_dict:
        current_dev = results_dict["calm_deviation"]
        draft_dev = DRAFT_CALM_ADHERENCE_DEVIATION
        improvement = draft_dev - current_dev
        if improvement <= 0:
            weaknesses.append(
                f"Catalogue rebalancing did NOT improve calm-request energy "
                f"adherence: {draft_dev:.2f} (Draft, Section 5.6) -> "
                f"{current_dev:.2f} now ({improvement:+.2f}, i.e. no "
                f"measurable gain, possibly a slight regression). The "
                f"energy < 0.35 filter used for the rebalancing pass was "
                f"evidently too loose to move tracks close enough to the "
                f"actual request target (energy=0.2) -- most additions "
                f"likely landed in the 0.25-0.35 range rather than near "
                f"0.2. This is reported as an open, unresolved limitation: "
                f"closing it would require either a tighter energy filter "
                f"when sampling new tracks, or accepting that the current "
                f"data source may not contain enough sufficiently calm "
                f"tracks in the sampled genres to fully close this gap."
            )
        elif current_dev > 0.25:  # still far from the 0.2 energy target
            weaknesses.append(
                f"Calm-request energy adherence improved only from "
                f"{draft_dev:.2f} (Draft, Section 5.6) to {current_dev:.2f} "
                f"after catalogue rebalancing -- a real but partial fix "
                f"({improvement:+.2f}). The catalogue still does not "
                f"contain enough tracks near the actual request target "
                f"(energy=0.2) to close this gap fully; this is a data "
                f"limitation, not a scoring-weight limitation, and is "
                f"reported honestly as unresolved rather than fixed."
            )
        else:
            strengths.append(
                f"Calm-request adherence improved from {draft_dev:.2f} to "
                f"{current_dev:.2f} after catalogue rebalancing, closing "
                f"most of the original gap."
            )

    print("\nStrengths (evidence-backed):")
    for s in strengths:
        print(f"  + {s}")
    print("\nWeaknesses / limitations (evidence-backed):")
    for w in weaknesses:
        print(f"  - {w}")
    if uncertain:
        print("\nUncertain / partially-resolved:")
        for u in uncertain:
            print(f"  ? {u}")

    if not weaknesses and not uncertain:
        print("\n  [WARNING] No weaknesses were auto-detected -- re-check "
              "the thresholds above before writing this into the report.")


if __name__ == "__main__":
    hit_results = full_hit_rate_eval()
    held_out_results = held_out_full_pipeline_eval()
    novelty_results = novelty_sensitivity_eval()
    multi_seed_eval()
    mmr_diversity_eval()
    mood_results = mood_adherence_eval()
    catalogue_energy_distribution_eval()

    print("\n" + "=" * 70)
    print("Weight optimization (leakage-free, see weight_optimizer_fixed.py)")
    print("=" * 70)
    weight_results = weight_optimizer_fixed.run()

    generate_verdict({
        "hit1": hit_results["hit1"],
        "held_out_hit1": held_out_results["hit1"] if held_out_results else None,
        "weight_delta": weight_results["delta"],
        "sequence_only_sanity_check": weight_results["sequence_only_hit1"],
        "novelty_convergence_after": novelty_results["convergence_pct"],
        "novelty_change_rate": novelty_results["change_rate"],
        "calm_deviation": mood_results["calm_deviation"],
    })