"""
merge_catalog.py
-------------------
Combines new_track_catalog.py (real audio features, 180 tracks) and
transitions.json (real playlist transitions) into a final track_catalog.py
that's a drop-in replacement for the original mock version -- same three
exported names (TRACK_CATALOG, FALLBACK_TAGS, TRANSITION_COUNTS), so
data.py and scorer.py need no changes.

Usage (run in the same folder as new_track_catalog.py and transitions.json):
    python merge_catalog.py
"""

import importlib.util
import json
import re
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# The original curated 2-tag fallbacks for the 14 tracks that were in the
# prototype from the start -- kept as-is, since they're already accurate
# and will be overwritten by real MusicBrainz tags once fetch_tags.py runs
# anyway.
ORIGINAL_FALLBACK_TAGS = {
    "t01": ["indie rock", "rock"], "t02": ["indie rock", "rock"],
    "t09": ["indie rock", "rock"], "t03": ["electronic", "house"],
    "t04": ["electronic", "house"], "t13": ["electronic", "house"],
    "t05": ["folk", "acoustic"], "t06": ["folk", "acoustic"],
    "t12": ["folk", "acoustic"], "t07": ["hip hop", "rap"],
    "t08": ["hip hop", "rap"], "t14": ["hip hop", "rap"],
    "t10": ["pop", "dance"], "t11": ["pop", "dance"],
}

def _normalize_title(title):
    # Same normalization scorer.py uses for seed/candidate matching --
    # strips " - 2018 Remaster", " (Extra Sped Up Version)", etc.
    return re.sub(r"\s*[-(].*$", "", title).strip().lower()
 
 
def _dedupe_catalog(catalog):
    """Collapses tracks that are the same underlying song under
    different release labels (remaster, sped-up, extended, etc.),
    keeping the highest-popularity version of each."""
    seen = {}
    dropped = []
    for track in catalog:
        key = _normalize_title(track["title"]) + "::" + track["artist"].lower()
        if key not in seen:
            seen[key] = track
        elif track["popularity"] > seen[key]["popularity"]:
            dropped.append(seen[key])
            seen[key] = track
        else:
            dropped.append(track)
    if dropped:
        print(f"Deduped {len(dropped)} near-identical track(s) (same "
              f"song, different release), keeping the higher-popularity "
              f"version of each:")
        for t in dropped:
            print(f"  removed: {t['id']} {t['artist']} - {t['title']} "
                  f"(popularity={t['popularity']})")
    return list(seen.values())

def load_catalog(path=None):
    path = path or os.path.join(_HERE, "new_track_catalog.py")
    spec = importlib.util.spec_from_file_location("new_track_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRACK_CATALOG

def main():
    catalog = load_catalog()
    catalog = _dedupe_catalog(catalog) 

    with open(os.path.join(_HERE, "transitions.json"), encoding="utf-8") as f:
        raw_transitions = json.load(f)
    transitions = {}
    for key, count in raw_transitions.items():
        a, b = key.split("::")
        transitions[(a, b)] = count

    fallback_tags = {}
    for track in catalog:
        tid = track["id"]
        if tid in ORIGINAL_FALLBACK_TAGS:
            fallback_tags[tid] = ORIGINAL_FALLBACK_TAGS[tid]
        else:
            # Placeholder until fetch_tags.py pulls real MusicBrainz tags --
            # a single genre-derived tag is enough for the scorer to run.
            fallback_tags[tid] = [track.get("genre_hint", "unknown")]

    with open(os.path.join(_ROOT, "track_catalog.py"), "w", encoding="utf-8") as out:
        out.write('"""\n')
        out.write("track_catalog.py\n")
        out.write("-----------------\n")
        out.write("The catalogue: 180 tracks. 10 of the original 14 tracks have real\n")
        out.write("Kaggle audio-feature values; 4 (t03, t04, t12, t13, t14) could not be\n")
        out.write("matched in the dataset and keep their original reasoned estimates,\n")
        out.write('explicitly flagged with "estimated": true. The 165 additional tracks\n')
        out.write("all have real audio features, sampled across 12 genres.\n\n")
        out.write("TRANSITION_COUNTS below are real adjacent-track co-occurrence counts\n")
        out.write("mined from the Kaggle \"Spotify Playlists\" dataset (~12.9M rows),\n")
        out.write("replacing the original fully-mock transition data.\n")
        out.write('"""\n\n')

        def py_repr(v):
            # json.dumps renders True/False as lowercase true/false, which is
            # valid JSON but invalid Python -- use repr() for bools instead.
            if isinstance(v, bool):
                return repr(v)
            return json.dumps(v)

        out.write("TRACK_CATALOG = [\n")
        for t in catalog:
            fields = ", ".join(f'"{k}": {py_repr(v)}' for k, v in t.items())
            out.write(f"    {{{fields}}},\n")
        out.write("]\n\n")

        out.write("# Fallback tags -- used only if MusicBrainz has no usable tags for a\n")
        out.write("# track. Original 14 keep their curated pairs; new tracks use a\n")
        out.write("# genre-derived placeholder until fetch_tags.py is re-run.\n")
        out.write("FALLBACK_TAGS = {\n")
        for tid, tags in fallback_tags.items():
            out.write(f"    {json.dumps(tid)}: {json.dumps(tags)},\n")
        out.write("}\n\n")

        out.write("# Real adjacent-track transition counts, mined from real playlists.\n")
        out.write("TRANSITION_COUNTS = {\n")
        for (a, b), count in transitions.items():
            out.write(f"    ({json.dumps(a)}, {json.dumps(b)}): {count},\n")
        out.write("}\n")

    print(f"Wrote track_catalog.py: {len(catalog)} tracks, "
          f"{len(fallback_tags)} fallback entries, {len(transitions)} real transitions.")
    print("Next: python fetch_tags.py   (to get real MusicBrainz tags for the new tracks)")


if __name__ == "__main__":
    main()