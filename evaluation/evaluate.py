"""
evaluate.py
------------
Offline evaluation of NextTrack's hybrid scorer, using the real
adjacent-track transitions mined from real playlists (extract_transitions.py).

Two things are measured:

  1. Hit-rate: for every catalogue track that has at least one real
     "what came next" transition, does the system's recommendation
     match a real next track? Compared against a random baseline, this
     is the accuracy metric promised in the report's evaluation plan
     (Section 3.8), now measurable because TRANSITION_COUNTS is real
     data instead of a mock.

  2. Novelty sensitivity: across a sample of seeds, how often does
     moving novelty from low to high actually change the top pick?
     This turns the Case 2 observation in Chapter 4 (novelty=0.9 still
     recommending the same familiar track) into a systematic, reportable
     number instead of a single anecdote.

Run with:
    python evaluate.py
"""

import random
from collections import defaultdict, Counter

import itertools
from data import TRACKS, TRACKS_BY_ID, TRANSITION_COUNTS
from scorer import recommend_next_track, audio_similarity

random.seed(42)


def real_next_tracks():
    """Maps each track id to the set of tracks that are known to have
    really followed it, according to the mined playlist data."""
    next_map = defaultdict(set)
    for (a, b) in TRANSITION_COUNTS:
        next_map[a].add(b)
    return next_map


def hit_rate_eval(top_n=5, sample_size=100):
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
    all_ids = [t["id"] for t in TRACKS]
    random.shuffle(all_ids)
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
    print(f"\nNovelty sensitivity: top pick changed between novelty=0.1 "
          f"and novelty=0.9 for {changed}/{n} seeds ({changed / n:.1%}).")
    if examples:
        print("  Examples where it did change:")
        for seed, low, high in examples:
            seed_t = TRACKS_BY_ID[seed]
            print(f"    {seed_t['title']} -> low: "
                  f"{TRACKS_BY_ID[low]['title']}, high: {TRACKS_BY_ID[high]['title']}")

    # Diagnostic: does novelty spread across many different tracks, or
    # does it collapse onto one or two "universal" low-popularity picks
    # regardless of the seed's genre?
    high_picks = [recommend_next_track([s], novelty=0.9, top_n=1)[0]["track_id"] for s in seeds]
    distinct = len(set(high_picks))
    most_common_id, most_common_count = Counter(high_picks).most_common(1)[0]
    print(f"\nHigh-novelty pick diversity: {distinct} distinct tracks recommended "
          f"across {n} different seeds.")
    print(f"  Most frequent high-novelty pick: "
          f"{TRACKS_BY_ID[most_common_id]['title']} "
          f"({most_common_count}/{n} = {most_common_count / n:.1%} of all seeds)")
    if most_common_count / n > 0.15:
        print("  -> Convergence: one low-popularity track is being boosted "
              "across many unrelated seeds, not genuine per-seed discovery.")


def multi_seed_eval(sample_size=60, history_len=3):
    """
    Checks whether using a full listening history (instead of just the
    last track) actually changes the recommendation, and how often.
    Builds synthetic short "sessions" by taking history_len consecutive
    tracks from the catalogue as a stand-in for a real session, since
    we don't have ground-truth multi-track sessions to sample from.

    Compares:
      - last-track-only recommendation:  recommend_next_track([last])
      - full-history recommendation:     recommend_next_track(history)

    A high rate of disagreement is evidence the recency-weighted
    multi-seed logic is doing real work, not just matching the old
    single-track behaviour.
    """
    all_ids = [t["id"] for t in TRACKS]
    import random
    random.seed(7)
    random.shuffle(all_ids)

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

    print(f"\nMulti-seed effect: recommendation changed between "
          f"last-track-only and full-{history_len}-track-history for "
          f"{changed}/{n} sessions ({changed / n:.1%}).")


def mmr_diversity_eval(sample_size=40, top_n=5):
    """
    Measures how audio-similar the tracks within a single /recommend/queue
    response are to each other. Lower average pairwise similarity means
    the MMR step is doing its job (returning a spread of tracks, not
    near-duplicates of the single best match).
    """
    all_ids = [t["id"] for t in TRACKS]
    import random
    random.seed(11)
    random.shuffle(all_ids)
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

def mood_adherence_eval(sample_size=40):
    """
    Checks whether the recommended track's actual energy value moves
    toward the requested energy target, comparing a "calm/low energy"
    request against an "energetic/high energy" request for the same
    seed. This is the direct evaluation of promoting context (mood +
    energy) from a +/-15% score multiplier to a first-class weighted
    term (WEIGHT_CONTEXT) in hybrid_score().
 
    Reports the average absolute difference between the requested
    energy and the recommended track's actual energy, for both a low-
    and a high-energy request. Lower = the system is adhering more
    closely to what was actually asked for.
    """
    all_ids = [t["id"] for t in TRACKS]
    random.shuffle(all_ids)
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

def catalogue_energy_distribution_eval():
    """
    Diagnostic for the calm/low-energy vs energetic/high-energy
    adherence gap seen in mood_adherence_eval(). If the catalogue is
    skewed toward higher-energy tracks, low-energy requests will
    always struggle to find a good match regardless of how much
    WEIGHT_CONTEXT is increased -- this tells that whether that's the
    actual cause before spend time retuning weights.
    """
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
        print("  -> Catalogue is skewed toward higher-energy tracks; this "
              "likely explains why calm/low-energy requests adhere less "
              "well than energetic/high-energy ones, independent of "
              "scoring weights.")

        
if __name__ == "__main__":
    hit_rate_eval()
    novelty_sensitivity_eval()
    multi_seed_eval()
    mmr_diversity_eval()
    mood_adherence_eval()
    catalogue_energy_distribution_eval()
