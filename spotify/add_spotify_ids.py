"""
add_spotify_ids.py
--------------------
Adds real Spotify track IDs to track_catalog.py by matching against the
Kaggle "Spotify Tracks Dataset" CSV downloaded for
build_catalog.py -- that dataset's "track_id" column IS the real
Spotify track ID for each row, since that's how Spotify's own export
identifies tracks. This avoids the Spotify Web API entirely, so it
doesn't run into the February 2026 Premium-account requirement for
registering a developer app.

Usage (run in the project folder, where track_catalog.py and
spotify_tracks_dataset.csv both already exist):
    python add_spotify_ids.py spotify_tracks_dataset.csv
"""

import csv
import importlib.util
import json
import sys


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()


def pop_of(row):
    try:
        return float(row.get("popularity", 0))
    except (TypeError, ValueError):
        return 0.0


def load_index(csv_path):
    index = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = norm(row["artists"].split(";")[0]) + "::" + norm(row["track_name"])
            if key not in index or pop_of(row) > pop_of(index[key]):
                index[key] = row
    return index


def load_current_catalog():
    spec = importlib.util.spec_from_file_location("current_catalog", "track_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRACK_CATALOG, mod.FALLBACK_TAGS, mod.TRANSITION_COUNTS


def py_repr(v):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return repr(v)
    return json.dumps(v)


def main():
    if len(sys.argv) != 2:
        print("Usage: python add_spotify_ids.py <spotify_tracks_dataset.csv>")
        sys.exit(1)

    index = load_index(sys.argv[1])
    catalog, fallback_tags, transitions = load_current_catalog()

    found = missed = 0
    for track in catalog:
        key = norm(track["artist"]) + "::" + norm(track["title"])
        row = index.get(key)
        if row:
            track["spotify_id"] = row["track_id"]
            found += 1
        else:
            track["spotify_id"] = None
            missed += 1

    with open("track_catalog.py", "w", encoding="utf-8") as out:
        out.write('"""\ntrack_catalog.py -- now includes real spotify_id for embedding playback.\n"""\n\n')
        out.write("TRACK_CATALOG = [\n")
        for t in catalog:
            fields = ", ".join(f'"{k}": {py_repr(v)}' for k, v in t.items())
            out.write(f"    {{{fields}}},\n")
        out.write("]\n\n")

        out.write("FALLBACK_TAGS = {\n")
        for tid, tags in fallback_tags.items():
            out.write(f"    {json.dumps(tid)}: {json.dumps(tags)},\n")
        out.write("}\n\n")

        out.write("TRANSITION_COUNTS = {\n")
        for (a, b), count in transitions.items():
            out.write(f"    ({json.dumps(a)}, {json.dumps(b)}): {count},\n")
        out.write("}\n")

    print(f"Added real Spotify IDs to {found}/{len(catalog)} tracks "
          f"({missed} missed -- these will have spotify_id: null, and the "
          f"frontend should fall back to a plain search link for them).")
    print("track_catalog.py rewritten in place.")


if __name__ == "__main__":
    main()
