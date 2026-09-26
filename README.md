# NextTrack — a stateless, privacy-aware music recommendation API

A RESTful "next track" recommendation system. No accounts, no stored
listening history — every request is self-contained: submit a short
sequence of recent track IDs plus mood/energy/novelty/diversity
preferences, and get back a recommended next track with a confidence
score and a plain-English explanation.

Built for CM3070 Template 7.2 (NextTrack: A music recommendation API).

## Project structure

```
nexttrack_prototype/
├── app.py                    # Flask API + Swagger docs (/apidocs)
├── data.py                   # builds TRACKS from track_catalog.py + real MusicBrainz tags
├── scorer.py                 # hybrid scoring, novelty re-ranking, MMR diversification
├── rate_limit_store.py       # persistent SQLite-backed rate limiter
├── track_catalog.py          # the built catalogue (generated — see "Rebuilding")
├── transitions.json          # real mined playlist transitions (generated)
├── templates/
│   └── index.html            # interactive web demo
├── datasets/                 # raw Kaggle CSVs 
├── catalog_pipeline/         # scripts that build/update track_catalog.py
│   ├── build_catalog.py
│   ├── merge_catalog.py
│   ├── extract_transitions.py
│   ├── rebalance_catalogue.py
│   ├── append_low_energy.py
│   ├── low_energy_additions.py    # generated
│   └── new_track_catalog.py       # generated
├── musicbrainz/               # tag fetching (offline setup step)
│   ├── musicbrainz_client.py
│   ├── fetch_tags.py
│   └── musicbrainz_cache.json     # generated
├── spotify/
│   └── add_spotify_ids.py     # attaches real Spotify IDs for embedded playback
├── weight_optimization/       # cross-validated weight search (see "Weight optimization")
│   ├── weight_optimizer_fixed.py  # leakage-free — trust this one
│   └── weight_optimizer.py        # superseded, kept for the report's methodology narrative
├── diagnostics/
│   └── check_mmr_variance.py  # quick sanity check that MMR actually differentiates
├── evaluation/
│   └── evaluate.py            # the full evaluation suite (see "Testing and evaluation")
├── tests/
│   ├── test_api.py
│   ├── test_scorer.py
│   └── test_prototype.py
├── requirements.txt
└── README.md
```

Root-level files (`app.py`, `data.py`, `scorer.py`, `rate_limit_store.py`,
`track_catalog.py`) are the live runtime import chain and stay at the
project root deliberately. Everything else is either a one-off/offline
script or reference material, grouped by purpose.

## Quickstart

```bash
# create and activate a virtual environment
py -m venv my_env
my_env\Scripts\activate          # Windows (PowerShell)
# source my_env/bin/activate     # macOS/Linux

pip install -r requirements.txt
python app.py
```

> **Windows PowerShell note:** if `activate` is blocked by the execution
> policy, run this once first: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Then open:
- **http://127.0.0.1:5000** — interactive web demo (see "The web demo" below)
- **http://127.0.0.1:5000/apidocs** — Swagger UI (interactive API docs)

Or call the API directly:

```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"track_ids": ["t01"], "mood": "energetic", "energy": 0.8, "novelty": 0.6}'
```

## How recommendations are made

Each candidate track is scored against the **whole submitted listening
history** (not just the last track), with more recent tracks weighted
more heavily via exponential recency decay:

| Component | Weight | What it measures |
|---|---|---|
| Audio similarity | 0.35 | Tempo/energy/valence distance to the history |
| Tag overlap | 0.25 | Shared MusicBrainz genre/style tags |
| Sequence | 0.20 | Real co-occurrence in mined playlist data |
| Context | 0.20 | Match to the requested mood/energy |

The base score is then re-ranked for **novelty** (surfaces less popular
tracks that are still related to the history, gated by a relatedness
floor so unrelated-but-rare tracks are never boosted) and, for
`/recommend/queue`, diversified across the returned set using
**Maximal Marginal Relevance** (Carbonell & Goldstein, 1998), so a
multi-track response isn't several near-duplicates of the top pick.

`/recommend`'s single top pick and `/recommend/queue`'s alternatives
are deliberately independent: the top pick is always the single
highest-scoring match and is never affected by the diversity
parameter, by design.

## The web demo

`templates/index.html` is a self-contained interactive client for the
API (no build step — it's plain HTML/CSS/JS served by Flask):

- **Ordered listening queue** — up to five seed tracks, shown with
  explicit position numbers and reorder controls, since the scorer's
  recency weighting depends on *order*, not just membership.
- **Mood presets** as quick-select chips, alongside manual energy/
  novelty/diversity sliders.
- **Now Playing** — the top recommendation with an animated confidence
  ring, a per-signal score breakdown, real MusicBrainz tags, and an
  embedded Spotify player (falls back to a search link for the small
  number of tracks with no matched Spotify ID).
- **More like this** — the MMR-diversified alternatives from
  `/recommend/queue`. Each row can be **played inline** (expands its
  own compact Spotify embed) or **queued**, which appends it to the
  listening order and lets you chain a multi-step session instead of
  firing one-shot requests.
- Debounced live search (`/search`, 300ms), with client-side fallback
  filtering if the endpoint is briefly unreachable.
- Keyboard-accessible throughout (visible focus states, ARIA roles on
  the track picker), and respects `prefers-reduced-motion`.

## Data sources

- **MusicBrainz** — real genre/style tags, fetched live and cached
  locally (`musicbrainz/fetch_tags.py` → `musicbrainz/musicbrainz_cache.json`).
  MusicBrainz's community tags are **not stable across fetches** — the
  same track can return different tags on different days — so treat one
  fetch run as a frozen snapshot rather than a value to keep regenerating.
- **Kaggle Spotify Tracks Dataset** — real audio features (tempo,
  energy, valence) and Spotify track IDs for embedded playback. Note:
  Spotify's own `audio-features`/`audio-analysis` API endpoints are
  permanently inaccessible for apps created after 27 November 2024,
  which is why this project uses the pre-existing Kaggle dataset
  instead of calling Spotify directly.
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Kaggle Spotify Playlists Dataset** — real adjacent-track
  transitions mined from ~12.9M playlist rows, replacing the original
  mock transition data.
  https://www.kaggle.com/datasets/andrewmvd/spotify-playlists

## Rebuilding the catalogue from scratch

Only needed if regenerating the dataset — the shipped
`track_catalog.py` / `musicbrainz/musicbrainz_cache.json` /
`transitions.json` already contain a built catalogue. Run from the
project root, in order:

```bash
# 1. Build base catalogue with real Kaggle audio features
python catalog_pipeline\build_catalog.py datasets\spotify_tracks_dataset.csv

# 2. Mine real transitions from the playlists dataset
python catalog_pipeline\extract_transitions.py datasets\spotify_dataset.csv catalog_pipeline\new_track_catalog.py

# 3. Merge into the final track_catalog.py at project root
python catalog_pipeline\merge_catalog.py

# 4. Fetch real MusicBrainz tags (needs internet, ~1-2 min, rate-limited to ~1 req/s)
python -m musicbrainz.fetch_tags

# 5. Attach real Spotify IDs for playback embeds
python spotify\add_spotify_ids.py datasets\spotify_tracks_dataset.csv
```

To also fix the catalogue's energy skew (see "Known limitations"):

```bash
python catalog_pipeline\rebalance_catalogue.py datasets\spotify_tracks_dataset.csv
python catalog_pipeline\append_low_energy.py
python -m musicbrainz.fetch_tags
python spotify\add_spotify_ids.py datasets\spotify_tracks_dataset.csv
```

macOS/Linux: replace `\` with `/` in the paths above.

## Weight optimization

`weight_optimization/weight_optimizer_fixed.py` cross-validates the
hand-chosen scoring weights against a genuinely held-out split of the
real transition data. **Use this script's numbers, not
`weight_optimizer.py`'s** — the older script has a circular-evaluation
bug (its ground truth and its sequence signal are both built from the
same unsplit `TRANSITION_COUNTS`, so sequence-only scoring trivially
hits 100%). `weight_optimizer.py` is kept only so the report can
describe the bug and its fix as a methodology finding; its own
headline numbers should never be cited as results.

```bash
python -m weight_optimization.weight_optimizer_fixed
```

## Testing and evaluation

```bash
python -m pytest tests/ -v
python -m evaluation.evaluate
python -m diagnostics.check_mmr_variance
```

`evaluation/evaluate.py` reports, against real mined playlist data:

- **Hit@1/Hit@5** — both a full leave-one-out figure over every
  eligible seed (with bootstrap 95% CIs) *and* a genuinely held-out
  figure computed with `TRANSITION_COUNTS` split before scoring. The
  gap between the two is the size of the sequence-signal leak in the
  in-sample number — report the held-out figure as the primary result.
- **Novelty sensitivity** — how often the top pick changes between
  low and high novelty, and whether high-novelty picks converge on a
  small set of "safe rare" tracks rather than genuinely varying per seed.
- **Multi-seed history effect** — how often using the full submitted
  sequence (vs. only the last track) changes the recommendation.
- **MMR diversity impact** — average pairwise audio similarity within
  a multi-track response, with vs. without diversification.
- **Mood/energy adherence** and **catalogue energy-distribution**
  diagnostics, to separate a scoring-weight limitation from a
  data/catalogue-composition limitation.

`generate_verdict()` at the end of the script turns all of the above
into an evidence-backed strengths/weaknesses readout, comparing each
current result against the fixed historical baselines from earlier in
the project (not re-derived, since they describe a system state that
no longer exists after subsequent fixes).

## Known limitations

- **Rate limiting** is backed by a local SQLite file
  (`rate_limit_store.py`), replacing Flask-Limiter's in-memory default
  so counts survive a server restart and are shared correctly across
  multiple worker processes *on the same machine*. It is still not a
  distributed solution — a real multi-machine deployment would need a
  shared store such as Redis.
- The catalogue skews toward higher-energy tracks, which measurably
  reduces adherence for calm/low-energy requests even after rebalancing
  (see `catalogue_energy_distribution_eval()` and `mood_adherence_eval()`
  in `evaluation/evaluate.py`) — a data-composition limitation, not a
  scoring-weight one.
- Novelty re-ranking can still converge on a small handful of "safe
  rare" tracks across otherwise unrelated seeds; the squared-
  relatedness fix reduces this but does not eliminate it, and trades
  off against how often novelty changes the top pick at all (see
  `novelty_sensitivity_eval()`).
- `/search` does a linear substring scan; fine at the current catalogue
  size, would need indexing at a much larger scale.
- MusicBrainz tag instability (see "Data sources") means re-running
  `fetch_tags.py` can shift reported tag-source and tag-overlap numbers
  between runs, even with an unchanged catalogue.