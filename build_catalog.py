"""
build_catalog.py
-----------------
Replaces NextTrack's estimated audio features with real values from the
Kaggle "Spotify Tracks Dataset" (maharshipandya/-spotify-tracks-dataset),
and scales the 14-track catalogue up to a larger, genre-diverse set.

Usage:
    python build_catalog.py spotify_tracks_dataset.csv

Expected input columns (this dataset's actual schema):
    track_id, artists, album_name, track_name, popularity, duration_ms,
    explicit, danceability, energy, key, loudness, mode, speechiness,
    acousticness, instrumentalness, liveness, valence, tempo,
    time_signature, track_genre

Output:
    new_track_catalog.py  -- a drop-in replacement body for TRACK_CATALOG
                              (paste into track_catalog.py, or import it)
"""

import csv
import difflib
import sys

# The 14 tracks already in the catalogue -- these get matched to real
# audio-feature rows rather than dropped, so Chapter 4's existing test
# cases and figures stay valid.
EXISTING_TRACKS = [
    ("t01", "Mr. Brightside", "The Killers"),
    ("t02", "Somebody Told Me", "The Killers"),
    ("t09", "When You Were Young", "The Killers"),
    ("t03", "One More Time", "Daft Punk"),
    ("t04", "Harder, Better, Faster, Stronger", "Daft Punk"),
    ("t13", "Digital Love", "Daft Punk"),
    ("t05", "Skinny Love", "Bon Iver"),
    ("t06", "Holocene", "Bon Iver"),
    ("t12", "Re: Stacks", "Bon Iver"),
    ("t07", "HUMBLE.", "Kendrick Lamar"),
    ("t08", "DNA.", "Kendrick Lamar"),
    ("t14", "Money Trees", "Kendrick Lamar"),
    ("t10", "Levitating", "Dua Lipa"),
    ("t11", "Don't Start Now", "Dua Lipa"),
]

# Genres to pull extra tracks from, to both reinforce the existing
# clusters and add new ones (widens novelty re-ranking testing).
GENRES_TO_SAMPLE = {
    "rock": 15, "alt-rock": 15, "house": 15, "electronic": 15,
    "folk": 15, "acoustic": 15, "hip-hop": 15, "pop": 15,
    "jazz": 15, "classical": 10, "r-n-b": 10, "indie": 10,
}
PER_GENRE_TOP_N = 30  # sample from top-N most popular rows per genre


# Fallback (estimated) values, used only if a track can't be matched to a
# real row at all -- kept explicit here so misses are never silently
# dropped from the catalogue.
ESTIMATED_FALLBACK = {
    "t01": (148, 0.82, 0.55, 0.95), "t02": (142, 0.85, 0.60, 0.80),
    "t09": (134, 0.78, 0.50, 0.78), "t03": (123, 0.90, 0.85, 0.90),
    "t04": (123, 0.88, 0.65, 0.85), "t13": (120, 0.70, 0.80, 0.70),
    "t05": (84, 0.30, 0.30, 0.75), "t06": (92, 0.25, 0.35, 0.65),
    "t12": (80, 0.20, 0.30, 0.45), "t07": (150, 0.80, 0.45, 0.92),
    "t08": (140, 0.88, 0.40, 0.85), "t14": (130, 0.55, 0.45, 0.80),
    "t10": (103, 0.80, 0.90, 0.93), "t11": (124, 0.85, 0.75, 0.90),
}


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pop_of(row):
    try:
        return float(row.get("popularity", 0))
    except (TypeError, ValueError):
        return 0.0


def match_existing(rows):
    index = {}
    for r in rows:
        key = norm(r["artists"].split(";")[0]) + "::" + norm(r["track_name"])
        if key not in index or pop_of(r) > pop_of(index[key]):
            index[key] = r

    matched, misses = [], []
    for tid, title, artist in EXISTING_TRACKS:
        key = norm(artist) + "::" + norm(title)
        row = index.get(key)
        match_type = "exact"
        if row is None:
            candidates = difflib.get_close_matches(key, index.keys(), n=1, cutoff=0.75)
            row = index[candidates[0]] if candidates else None
            match_type = "fuzzy"

        # NEW: reject fuzzy matches with suspiciously low popularity for
        # tracks known to be mainstream hits -- these are usually wrong
        # matches (remixes, edits, or unrelated tracks with similar names)
        # rather than genuinely obscure versions.
        if row and match_type == "fuzzy" and pop_of(row) < 20:
            print(f"  [rejected] {tid} {artist} - {title}: fuzzy match with "
                f"popularity={pop_of(row)} is implausible -- falling back "
                f"to estimated values instead of trusting this match")
            row = None
            match_type = "rejected"

        if row:
            matched.append((tid, title, artist, row))
        else:
            misses.append((tid, title, artist))
    return matched, misses


def sample_new_tracks(rows, existing_keys, start_num=15):
    by_genre = {}
    for r in rows:
        by_genre.setdefault(r["track_genre"], []).append(r)

    picked, next_id = [], start_num
    for genre, count in GENRES_TO_SAMPLE.items():
        pool = by_genre.get(genre, [])
        pool.sort(key=lambda r: float(r.get("popularity", 0)), reverse=True)
        added = 0
        for r in pool[:PER_GENRE_TOP_N]:
            key = norm(r["artists"].split(";")[0]) + "::" + norm(r["track_name"])
            if key in existing_keys or added >= count:
                continue
            existing_keys.add(key)
            picked.append((f"t{next_id:03d}", r))
            next_id += 1
            added += 1
    return picked


def fmt_row(tid, title, artist, row):
    tempo = round(float(row["tempo"]), 1)
    energy = round(float(row["energy"]), 2)
    valence = round(float(row["valence"]), 2)
    popularity = round(float(row.get("popularity", 0)) / 100.0, 2)
    title = title.replace('"', '\\"')
    artist = artist.replace('"', '\\"')
    genre = row.get("track_genre", "")
    return (f'    {{"id": "{tid}", "title": "{title}", "artist": "{artist}", '
            f'"tempo": {tempo}, "energy": {energy}, "valence": {valence}, '
            f'"popularity": {popularity}, "genre_hint": "{genre}"}},')


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_catalog.py <spotify_tracks_dataset.csv>")
        sys.exit(1)

    rows = load_rows(sys.argv[1])
    matched, misses = match_existing(rows)
    print(f"Matched {len(matched)}/{len(EXISTING_TRACKS)} existing tracks to real audio features.")
    if misses:
        print("Could not match -- kept as clearly-labelled estimated values:")
        for tid, title, artist in misses:
            print(f"  {tid}: {title} - {artist}")

    existing_keys = {norm(a) + "::" + norm(t) for _, t, a, _ in matched}
    existing_keys |= {norm(a) + "::" + norm(t) for _, t, a in misses}
    new_tracks = sample_new_tracks(rows, existing_keys)
    print(f"Sampled {len(new_tracks)} additional tracks across "
          f"{len(GENRES_TO_SAMPLE)} genres.")

    with open("new_track_catalog.py", "w", encoding="utf-8") as out:
        out.write("# Auto-generated by build_catalog.py -- real Kaggle audio features\n")
        out.write("# (\"estimated\": true rows kept the original reasoned estimates,\n")
        out.write("#  since no real match was found for them).\n")
        out.write("TRACK_CATALOG = [\n")
        for tid, title, artist, row in matched:
            out.write(fmt_row(tid, title, artist, row) + "\n")
        for tid, title, artist in misses:
            tempo, energy, valence, popularity = ESTIMATED_FALLBACK[tid]
            title_e, artist_e = title.replace('"', '\\"'), artist.replace('"', '\\"')
            out.write(f'    {{"id": "{tid}", "title": "{title_e}", "artist": "{artist_e}", '
                       f'"tempo": {tempo}, "energy": {energy}, "valence": {valence}, '
                       f'"popularity": {popularity}, "genre_hint": "unknown", '
                       f'"estimated": True}},\n')
        for tid, row in new_tracks:
            out.write(fmt_row(tid, row["track_name"], row["artists"].split(";")[0], row) + "\n")
        out.write("]\n")

    print(f"\nWrote new_track_catalog.py with "
          f"{len(matched) + len(misses) + len(new_tracks)} tracks total "
          f"({len(misses)} kept as estimated).")
    print("Next: paste this into track_catalog.py in place of TRACK_CATALOG, "
          "then re-run fetch_tags.py to get real MusicBrainz tags for the "
          "new tracks (FALLBACK_TAGS can use genre_hint until then).")


if __name__ == "__main__":
    main()