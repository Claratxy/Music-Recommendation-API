"""
musicbrainz_client.py
-----------------------
A small client for the MusicBrainz API, used to fetch real genre/style
tags for real tracks. This is the live "MusicBrainz metadata" data
source described in Chapter 3 of the Preliminary Report.

MusicBrainz is free and needs no API key, but its etiquette rules ask
every client to:
  1. Send a descriptive User-Agent header identifying the application.
  2. Make no more than 1 request per second.
(See: https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting)

This client also caches every lookup result to a local JSON file
(musicbrainz_cache.json), so the same track is never looked up twice,
and the rest of the project can run completely offline once the cache
has been built once.

Lesson learned from real testing: individual MusicBrainz "recording"
entries are often untagged, even for very famous songs, because
community tags are usually added to the ARTIST or the ALBUM, not to one
specific recording. So this client tries the recording first, and
falls back to the artist's own tags if the recording has none.
Folksonomy tags also include non-genre noise (e.g. "dolby atmos", a mix
format, not a genre), so a small filter removes obviously non-genre
tags before they are used.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error

BASE_URL = "https://musicbrainz.org/ws/2"

# IMPORTANT: MusicBrainz asks every client to identify itself with real contact detail
# fetch_tags.py for real.
USER_AGENT = "NextTrackPrototype/0.1 ( CM3070 student project; contact: xinyue274404@gmail.com )"

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "musicbrainz_cache.json")

# Tags that show up in MusicBrainz folksonomy data but are not genres or
# moods, so they would only add noise to tag-overlap scoring. This list
# is not exhaustive; it covers the patterns observed during real testing
# of this project, and may need extending if new junk tags turn up.
JUNK_TAGS = {
    "dolby atmos", "remaster", "remastered", "remastered version",
    "explicit", "clean", "clean version", "deluxe", "deluxe edition",
    "mono", "stereo", "vinyl", "cd single", "compilation",
    "greatest hits", "best of", "single", "soundtrack", "spotify",
    "itunes", "apple music",
}
_DECADE_OR_YEAR_RE = re.compile(r"^\d{4}$|^\d0s$|^\d{4}s$")


def _is_junk_tag(tag):
    if tag in JUNK_TAGS:
        return True
    if _DECADE_OR_YEAR_RE.match(tag):
        return True
    return False


_last_request_time = 0.0


def _rate_limited_get(url):
    """Performs a GET request, waiting if needed so that we never send
    more than roughly 1 request per second, as MusicBrainz's etiquette
    rules ask for."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    _last_request_time = time.time()
    return data


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    cache.setdefault("recordings", {})
    cache.setdefault("artists", {})
    return cache


def _save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _extract_tags(data):
    """Pulls tags+genres out of a MusicBrainz JSON entity, merges their
    vote counts, filters out junk tags, and returns them sorted by
    vote count (most-voted first)."""
    counted = {}
    for item in data.get("tags", []) + data.get("genres", []):
        name = item["name"].lower().strip()
        if _is_junk_tag(name):
            continue
        counted[name] = counted.get(name, 0) + item.get("count", 1)
    return [name for name, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)]


def search_recording_mbid(title, artist):
    """Looks up a recording's MBID by title and artist. Returns the
    MBID of the best-scoring match, or None if nothing reasonable was
    found."""
    query = f'recording:"{title}" AND artist:"{artist}"'
    url = f"{BASE_URL}/recording?query={urllib.parse.quote(query)}&fmt=json&limit=5"
    data = _rate_limited_get(url)

    recordings = data.get("recordings", [])
    if not recordings:
        return None
    best = max(recordings, key=lambda r: int(r.get("score", 0)))
    if int(best.get("score", 0)) < 50:
        return None
    return best["id"]


def search_artist_mbid(artist):
    """Looks up an artist's MBID by name. Returns the MBID of the
    best-scoring match, or None if nothing reasonable was found."""
    query = f'artist:"{artist}"'
    url = f"{BASE_URL}/artist?query={urllib.parse.quote(query)}&fmt=json&limit=5"
    data = _rate_limited_get(url)

    artists = data.get("artists", [])
    if not artists:
        return None
    best = max(artists, key=lambda a: int(a.get("score", 0)))
    if int(best.get("score", 0)) < 50:
        return None
    return best["id"]


def get_tags_for_recording(mbid):
    """Fetches filtered tags+genres for a recording MBID."""
    url = f"{BASE_URL}/recording/{mbid}?inc=tags+genres&fmt=json"
    return _extract_tags(_rate_limited_get(url))


def get_tags_for_artist(mbid):
    """Fetches filtered tags+genres for an artist MBID, using the
    on-disk cache (separate namespace from recordings, since many
    tracks share one artist)."""
    cache = _load_cache()
    if mbid in cache["artists"]:
        return cache["artists"][mbid]["tags"]

    url = f"{BASE_URL}/artist/{mbid}?inc=tags+genres&fmt=json"
    tags = _extract_tags(_rate_limited_get(url))

    cache["artists"][mbid] = {"tags": tags}
    _save_cache(cache)
    return tags


def get_tags_for_track(title, artist, max_tags=3):
    """
    High-level helper: given a track title and artist, returns up to
    max_tags real MusicBrainz tags/genres, plus a short string saying
    where they came from ("recording", "artist", or "none").

    Tries the recording first. If that has no usable tags, falls back
    to the artist's own tags, since MusicBrainz community tagging is
    much more consistent at the artist level than the individual
    recording level. Uses an on-disk cache throughout, so nothing is
    looked up twice.

    Returns ([], "none") rather than raising, if the track cannot be
    found or MusicBrainz cannot be reached, so a single bad lookup
    never crashes the whole fetch run.
    """
    cache = _load_cache()
    cache_key = f"{artist.lower()}::{title.lower()}"

    if cache_key in cache["recordings"]:
        entry = cache["recordings"][cache_key]
        return entry["tags"][:max_tags], entry["source"]

    recording_mbid = None
    tags = []
    source = "none"

    try:
        recording_mbid = search_recording_mbid(title, artist)
        if recording_mbid is not None:
            tags = get_tags_for_recording(recording_mbid)
            if tags:
                source = "recording"

        if not tags:
            artist_mbid = search_artist_mbid(artist)
            if artist_mbid is not None:
                tags = get_tags_for_artist(artist_mbid)
                if tags:
                    source = "artist"

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  [warning] MusicBrainz lookup failed for '{title}' by {artist}: {exc}")
        return [], "none"

    cache = _load_cache()  # reload, in case get_tags_for_artist() updated it
    cache["recordings"][cache_key] = {"mbid": recording_mbid, "tags": tags, "source": source}
    _save_cache(cache)
    return tags[:max_tags], source
