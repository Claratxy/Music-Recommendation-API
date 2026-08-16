"""
fetch_tags.py
--------------
One-time (or refresh-anytime) setup script: looks up real MusicBrainz
tags for every track in track_catalog.py, and saves them to
musicbrainz_cache.json.

Run with:
    python fetch_tags.py

This needs an internet connection and takes about 30-90 seconds (roughly
1 request per second, 1-2 requests per track, fewer once artist tags are
cached and shared across that artist's other tracks). After it has run
once, data.py loads everything from the cache file, so the rest of the
project (scorer.py, app.py, test_prototype.py) needs no network access
at all.
"""

from track_catalog import TRACK_CATALOG
from musicbrainz_client import get_tags_for_track

if __name__ == "__main__":
    print(f"Looking up {len(TRACK_CATALOG)} tracks on MusicBrainz...")
    print("(About 1 request per second, so this takes a minute or two.)\n")

    counts = {"recording": 0, "artist": 0, "none": 0}
    for track in TRACK_CATALOG:
        tags, source = get_tags_for_track(track["title"], track["artist"], max_tags=3)
        counts[source] += 1
        label = {"recording": "OK (track) ", "artist": "OK (artist)", "none": "MISS       "}[source]
        if tags:
            print(f"  {label} {track['artist']} - {track['title']}: {tags}")
        else:
            print(f"  {label} {track['artist']} - {track['title']}: no tags found, will use fallback")

    print(f"\nDone. {counts['recording']} from the recording itself, "
          f"{counts['artist']} from the artist, {counts['none']} missed completely "
          f"(out of {len(TRACK_CATALOG)}).")
    print("Results saved to musicbrainz_cache.json.")
