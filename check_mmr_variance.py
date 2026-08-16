"""
Run this once from the project root (same folder as data.py) to check
whether the catalogue has enough audio-feature variance for MMR to
produce visibly different top-4 sets at diversity=0 vs diversity=1.

    python check_mmr_variance.py
"""
from data import TRACKS_BY_ID
from scorer import recommend_next_track

seed = "t01"  # change to any real track_id in the catalogue

low = recommend_next_track([seed], top_n=4, diversity=0.0)
high = recommend_next_track([seed], top_n=4, diversity=1.0)

print("diversity=0.0:")
for r in low:
    print(f"  {r['track_id']}: {r['title']} (score={r['confidence_score']})")

print("\ndiversity=1.0:")
for r in high:
    print(f"  {r['track_id']}: {r['title']} (score={r['confidence_score']})")

low_ids = {r["track_id"] for r in low}
high_ids = {r["track_id"] for r in high}
overlap = low_ids & high_ids
print(f"\nOverlap: {len(overlap)}/4 tracks identical between diversity=0.0 and diversity=1.0")
if overlap == low_ids == high_ids:
    print("-> IDENTICAL results at both extremes. This means MMR is not "
          "differentiating at all for this seed -- either a code/server "
          "issue (restart Flask, check on the updated scorer.py), "
          "or this seed's candidate pool genuinely has too little "
          "audio-feature spread to matter.")
