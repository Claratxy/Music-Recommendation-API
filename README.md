# NextTrack — a stateless, privacy-aware music recommendation API

A RESTful "next track" recommendation system. No accounts, no stored
listening history — every request is self-contained: submit a short
sequence of recent track IDs plus mood/energy/novelty/diversity
preferences, and get back a recommended next track with a confidence
score and a plain-English explanation.

Built for CM3070 Template 7.2 (NextTrack: A music recommendation API).

## Quickstart

```bash
powershell -ExecutionPolicy Bypass
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

py -m venv my_env
my_env\Scripts\activate          # Windows
# source my_env/bin/activate     # macOS/Linux

pip install -r requirements.txt
py app.py
```

Then open:
- **http://127.0.0.1:5000** — interactive web demo
- **http://127.0.0.1:5000/apidocs** — Swagger UI (interactive API docs)

Or call the API directly:

```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"track_ids": ["t01"], "mood": "energetic", "energy": 0.8, "novelty": 0.6}'
```

## How recommendations are made

Each candidate track is scored against the whole listening history
(not just the last track), with more recent tracks weighted more
heavily:

| Component | Weight | What it measures |
|---|---|---|
| Audio similarity | 0.35 | Tempo/energy/valence distance to the history |
| Tag overlap | 0.25 | Shared MusicBrainz genre/style tags |
| Sequence | 0.20 | Real co-occurrence in mined playlist data |
| Context | 0.20 | Match to the requested mood/energy |

The base score is then re-ranked for **novelty** (surfaces less
popular tracks that are still related to the history) and, for
`/recommend/queue`, diversified across the returned set using
**Maximal Marginal Relevance** (Carbonell & Goldstein, 1998) so a
multi-track response isn't several near-duplicates of the top pick.

## Data sources

- **MusicBrainz** — real genre/style tags, fetched live and cached
  locally (`fetch_tags.py` → `musicbrainz_cache.json`)
- **Kaggle Spotify Tracks Dataset** — real audio features (tempo,
  energy, valence) and Spotify track IDs for embedded playback
- **Kaggle Spotify Playlists Dataset** — real adjacent-track
  transitions mined from ~12.9M playlist rows, replacing the original
  mock transition data

## Rebuilding the catalogue from scratch

Only needed if regenerating the dataset; the shipped
`track_catalog.py` / `musicbrainz_cache.json` / `transitions.json`
already contain the built catalogue.

```bash
python build_catalog.py datasets\spotify_tracks_dataset.csv
python extract_transitions.py datasets\spotify_dataset.csv new_track_catalog.py
python merge_catalog.py
python fetch_tags.py
python add_spotify_ids.py datasets\spotify_tracks_dataset.csv
```

## Testing and evaluation

```bash
python -m pytest tests/ -v
python -m evaluation.evaluate
```

`evaluation/evaluate.py` reports, against real mined playlist data:
Hit@1/Hit@5 vs. a random baseline, novelty sensitivity, multi-seed
history effect, MMR diversity impact, mood/energy adherence, and
catalogue energy-distribution diagnostics.

## Known limitations

- Rate limiting uses in-memory storage (`flask-limiter` default) —
  fine for a single-process demo, not safe for multiple workers.
- The catalogue skews toward higher-energy tracks, which measurably
  reduces adherence for calm/low-energy requests (see
  `catalogue_energy_distribution_eval()`).
- `/search` does a linear substring scan; fine at ~180 tracks, would
  need indexing at much larger catalogue sizes.
- The MMR-diversified alternatives list (`/recommend/queue`) and the
  single top pick (`/recommend`) are deliberately independent: the top
  pick is always the single highest-scoring match and is not affected
  by the diversity parameter, by design.
