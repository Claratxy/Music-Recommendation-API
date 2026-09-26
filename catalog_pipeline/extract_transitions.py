"""
extract_transitions.py
------------------------
Replaces NextTrack's mock TRANSITION_COUNTS with real adjacent-track
co-occurrence counts, mined from the Kaggle "Spotify Playlists" dataset
(andrewmvd/spotify-playlists). That file is ~1.2GB, so this script
streams it in chunks rather than loading it all into memory -- run it
locally on the raw download, not by uploading the full CSV.

Expected input columns (this dataset's actual schema):
    user_id, artistname, trackname, playlistname
Rows for the same playlist are assumed to appear in listening order,
which is the standard, documented way this dataset is used -- note this
as an accepted assumption/limitation in the evaluation chapter.

Usage:
    python extract_transitions.py spotify_dataset.csv new_track_catalog.py

Output:
    transitions.json  -- {"from_id::to_id": count, ...}
    This is small; upload just this file back.
"""

import csv
import json
import sys
import re
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

CHUNK_ROWS = 200_000


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()


def load_catalog_keys(catalog_path):
    """Parses the TRACK_CATALOG list out of a track_catalog.py-style file
    without importing it (so this script has no dependency on the rest
    of the project)."""
    text = open(catalog_path, encoding="utf-8").read()
    rows = re.findall(
        r'"id":\s*"([^"]+)".*?"title":\s*"([^"]+)".*?"artist":\s*"([^"]+)"',
        text,
    )
    keys = {}
    for tid, title, artist in rows:
        keys[norm(artist) + "::" + norm(title)] = tid
    return keys


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_transitions.py <playlists.csv> <track_catalog.py>")
        sys.exit(1)

    playlists_path, catalog_path = sys.argv[1], sys.argv[2]
    catalog_keys = load_catalog_keys(catalog_path)
    print(f"Loaded {len(catalog_keys)} catalogue tracks to match against.")

    transitions = {}
    prev_key = None
    prev_playlist = None
    rows_seen = 0

    with open(playlists_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            rows_seen += 1
            if rows_seen % CHUNK_ROWS == 0:
                print(f"  processed {rows_seen:,} rows...")

            playlist_id = (row.get("user_id", ""), row.get("playlistname", ""))
            key = norm(row.get("artistname", "")) + "::" + norm(row.get("trackname", ""))
            tid = catalog_keys.get(key)

            if playlist_id != prev_playlist:
                prev_key = None  # playlist boundary -- don't link across playlists
            prev_playlist = playlist_id

            if tid and prev_key:
                pair = f"{prev_key}::{tid}"
                if prev_key != tid:
                    transitions[pair] = transitions.get(pair, 0) + 1

            prev_key = tid if tid else prev_key

    with open(os.path.join(_HERE, "transitions.json"), "w", encoding="utf-8") as out:
        json.dump(transitions, out, indent=2, sort_keys=True)

    print(f"\nProcessed {rows_seen:,} rows total.")
    print(f"Found {len(transitions)} distinct real transitions between catalogue tracks.")
    print("Wrote transitions.json -- upload this file, and I'll fold it "
          "into TRANSITION_COUNTS in track_catalog.py.")


if __name__ == "__main__":
    main()
