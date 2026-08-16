"""
data.py
--------
Builds the final TRACKS / TRACKS_BY_ID / TRANSITION_COUNTS structures
that scorer.py uses, by combining:

  - track_catalog.py        -> real titles/artists + estimated audio features
  - musicbrainz_cache.json  -> REAL tags, fetched live from MusicBrainz by
                                fetch_tags.py (via musicbrainz_client.py)
  - FALLBACK_TAGS in track_catalog.py -> used only if a track has no
    cached MusicBrainz tags yet (e.g. fetch_tags.py has not been run, or
    that particular lookup found nothing)

This replaces the old, fully-mock mock_data.py. Run fetch_tags.py first
to populate real tags; if skipping that step, fallback tags are used
instead and the prototype still runs, just with that one part still
mocked, exactly like before.
"""

import json
import os

from track_catalog import TRACK_CATALOG, FALLBACK_TAGS, TRANSITION_COUNTS

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "musicbrainz_cache.json")

# musicbrainz_client.py caches the FULL filtered tag list for each track
# (sometimes 10-20+ tags for heavily-tagged artists), not just the first
# few. That is deliberate, so the cache stays useful even if this limit
# changes later. This is the one place that decides how many of those
# cached tags actually get used for scoring and shown in explanations.
MAX_TAGS_USED = 3


def _load_real_tags():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return cache.get("recordings", {})


def _build_tracks():
    real_tags = _load_real_tags()
    tracks = []

    for entry in TRACK_CATALOG:
        cache_key = f"{entry['artist'].lower()}::{entry['title'].lower()}"
        cached = real_tags.get(cache_key)

        if cached and cached.get("tags"):
            tags = set(cached["tags"][:MAX_TAGS_USED])
            source = f"musicbrainz ({cached.get('source', 'unknown')})"
        else:
            tags = set(FALLBACK_TAGS.get(entry["id"], []))
            source = "fallback"

        track = dict(entry)
        track["tags"] = tags
        track["tag_source"] = source  # handy to print/show in the demo video
        tracks.append(track)

    return tracks


TRACKS = _build_tracks()
TRACKS_BY_ID = {t["id"]: t for t in TRACKS}
# TRANSITION_COUNTS is re-exported here unchanged, so scorer.py's import
# line doesn't need to know it actually lives in track_catalog.py.