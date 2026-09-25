"""
app.py
-------
A small Flask app exposing the hybrid scorer as a REST API, plus a
simple web page (templates/index.html) that calls it and visualises
the result. This is the same API described in the project proposal and
Template 7.2; the web page is just a thin demo client on top of it.

Interactive API documentation (Swagger UI) is served at /apidocs once
this app is running, generated automatically from the docstrings below.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in a browser for the visual demo,
http://127.0.0.1:5000/apidocs for interactive API docs, or call the
API directly (in another terminal):
    curl -X POST http://127.0.0.1:5000/recommend \\
        -H "Content-Type: application/json" \\
        -d '{"track_ids": ["t01"], "mood": "energetic", "energy": 0.8, "novelty": 0.6}'
"""

from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
from rate_limit_store import rate_limited

from scorer import recommend_next_track
from data import TRACKS_BY_ID, TRACKS

app = Flask(__name__)

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "NextTrack API",
        "description": (
            "A stateless, privacy-aware music recommendation API. Submit a "
            "short list of recent track IDs plus mood/energy/novelty "
            "preferences, and get back a recommended next track with a "
            "confidence score, a plain-English explanation, and a "
            "playable Spotify embed. No accounts, no stored listening "
            "history -- every request is self-contained."
        ),
        "version": "1.0.0",
    },
    "basePath": "/",
}
swagger = Swagger(app, template=swagger_template)


def _validate_request(body):
    """Shared input validation for /recommend and /recommend/queue.
    Returns (track_ids, mood, energy, novelty, error_response) -- the
    first four are None if error_response is not None."""
    track_ids = body.get("track_ids", [])
    mood = body.get("mood", "neutral")
    energy = float(body.get("energy", 0.5))
    novelty = float(body.get("novelty", 0.3))

    if not track_ids or not (1 <= len(track_ids) <= 5):
        return None, None, None, None, (jsonify({"error": "track_ids must contain 1-5 track ids"}), 400)
    unknown_ids = [t for t in track_ids if t not in TRACKS_BY_ID]
    if unknown_ids:
        return None, None, None, None, (jsonify({"error": f"unknown track_ids: {unknown_ids}"}), 400)
    if not (0.0 <= energy <= 1.0) or not (0.0 <= novelty <= 1.0):
        return None, None, None, None, (jsonify({"error": "energy and novelty must be between 0.0 and 1.0"}), 400)

    return track_ids, mood, energy, novelty, None


@app.route("/")
def home():
    """Serves the demo web page."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check
    ---
    tags:
      - System
    responses:
      200:
        description: The API is up and the catalogue is loaded
        schema:
          type: object
          properties:
            status: {type: string, example: ok}
            tracks_loaded: {type: integer, example: 179}
    """
    return jsonify({"status": "ok", "tracks_loaded": len(TRACKS)})


@app.route("/tracks", methods=["GET"])
def list_tracks():
    """
    List every track in the catalogue
    ---
    tags:
      - Catalogue
    description: >
      Returns every track available for use as a seed track, sorted by
      artist then title. Intended for building a track picker UI.
    responses:
      200:
        description: All tracks in the catalogue
        schema:
          type: array
          items:
            type: object
            properties:
              id: {type: string, example: t01}
              title: {type: string, example: Mr. Brightside}
              artist: {type: string, example: The Killers}
    """
    return jsonify([
        {"id": t["id"], "title": t["title"], "artist": t["artist"]}
        for t in sorted(TRACKS, key=lambda t: (t["artist"], t["title"]))
    ])

@app.route("/search", methods=["GET"])
def search_tracks():
    """
    Search tracks by title or artist
    ---
    tags:
      - Catalogue
    parameters:
      - in: query
        name: q
        type: string
        required: true
        description: Substring to match against track title or artist (case-insensitive).
    description: >
      Case-insensitive substring search across title and artist.
      Returns at most 30 matches, sorted by artist then title. Intended
      to back a live search box in the track-picker UI so users aren't
      stuck scrolling the full catalogue.
    responses:
      200:
        description: Matching tracks (same shape as /tracks)
        schema:
          type: array
          items:
            type: object
            properties:
              id: {type: string, example: t01}
              title: {type: string, example: Mr. Brightside}
              artist: {type: string, example: The Killers}
      400:
        description: Missing or too-short q parameter
    """
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify({"error": "q query parameter is required"}), 400
    if len(query) < 2:
        return jsonify({"error": "q must be at least 2 characters"}), 400
 
    matches = [
        {"id": t["id"], "title": t["title"], "artist": t["artist"]}
        for t in TRACKS
        if query in t["title"].lower() or query in t["artist"].lower()
    ]
    matches.sort(key=lambda t: (t["artist"], t["title"]))
    return jsonify(matches[:30])

@app.route("/recommend", methods=["POST"])
@rate_limited(limit=20, window_seconds=60)
def recommend():
    """
    Recommend a single next track
    ---
    tags:
      - Recommendation
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [track_ids]
          properties:
            track_ids:
              type: array
              items: {type: string}
              minItems: 1
              maxItems: 5
              example: ["t01", "t02"]
              description: 1-5 recently played track IDs, most recent last.
            mood:
              type: string
              enum: [energetic, happy, calm, melancholy, neutral]
              default: neutral
            energy:
              type: number
              minimum: 0
              maximum: 1
              default: 0.5
            novelty:
              type: number
              minimum: 0
              maximum: 1
              default: 0.3
              description: 0 favours familiar tracks, 1 favours discovery.
    responses:
      200:
        description: A recommended next track
        schema:
          type: object
          properties:
            track_id: {type: string}
            title: {type: string}
            artist: {type: string}
            score: {type: number}
            reason: {type: string}
            spotify_id: {type: string}
            breakdown:
              type: object
              properties:
                audio: {type: number}
                tags: {type: number}
                sequence: {type: number}
                context: {type: number}
            tags:
              type: array
              items: {type: string}
            tag_source: {type: string}
      400:
        description: Invalid track_ids, energy, or novelty
      429:
        description: Rate limit exceeded (10 requests/minute)
    """
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "request body must be valid JSON"}), 400
    track_ids, mood, energy, novelty, error = _validate_request(body)
    if error:
        return error

    results = recommend_next_track(track_ids, mood=mood, energy=energy, novelty=novelty, top_n=1)
    top = results[0]

    return jsonify({
        "track_id": top["track_id"],
        "title": top["title"],
        "artist": top["artist"],
        "score": top["confidence_score"],
        "reason": top["reason"],
        "breakdown": top["breakdown"],
        "tags": top["tags"],
        "tag_source": top["tag_source"],
        "spotify_id": top.get("spotify_id"),
    })


@app.route("/recommend/queue", methods=["POST"])
@rate_limited(limit=20, window_seconds=60)
def recommend_queue():
    """
    Recommend several diverse next tracks
    ---
    tags:
      - Recommendation
    description: >
      Like /recommend, but returns several candidates using Maximal
      Marginal Relevance (Carbonell & Goldstein, 1998) instead of the
      plain top-N by score. This prevents a multi-track result from
      being several near-duplicates of the single best match -- each
      additional track is chosen to balance high relevance against low
      audio similarity to tracks already selected.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [track_ids]
          properties:
            track_ids:
              type: array
              items: {type: string}
              minItems: 1
              maxItems: 5
            mood:
              type: string
              enum: [energetic, happy, calm, melancholy, neutral]
              default: neutral
            energy: {type: number, minimum: 0, maximum: 1, default: 0.5}
            novelty: {type: number, minimum: 0, maximum: 1, default: 0.3}
            count:
              type: integer
              minimum: 2
              maximum: 5
              default: 3
              description: How many diverse tracks to return.
            diversity:
              type: number
              minimum: 0
              maximum: 1
              default: 0.3
              description: >
                0 favours the single most relevant tracks (may return
                near-duplicates). 1 favours spread, trading relevance
                for variety across the returned tracks. Implemented via
                Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
    responses:
      200:
        description: A list of diverse recommended tracks
        schema:
          type: object
          properties:
            recommendations:
              type: array
              items:
                type: object
                properties:
                  track_id: {type: string}
                  title: {type: string}
                  artist: {type: string}
                  score: {type: number}
                  reason: {type: string}
                  spotify_id: {type: string}
      400:
        description: Invalid track_ids, energy, novelty, or count
      429:
        description: Rate limit exceeded (10 requests/minute)
    """
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "request body must be valid JSON"}), 400
    track_ids, mood, energy, novelty, error = _validate_request(body)
    if error:
        return error

    count = int(body.get("count", 3))
    if not (2 <= count <= 5):
        return jsonify({"error": "count must be between 2 and 5"}), 400
 
    diversity = float(body.get("diversity", 0.3))
    if not (0.0 <= diversity <= 1.0):
        return jsonify({"error": "diversity must be between 0.0 and 1.0"}), 400
 
    results = recommend_next_track(
        track_ids, mood=mood, energy=energy, novelty=novelty,
        top_n=count, diversity=diversity,
    )

    return jsonify({
        "recommendations": [
            {
                "track_id": r["track_id"],
                "title": r["title"],
                "artist": r["artist"],
                "score": r["confidence_score"],
                "reason": r["reason"],
                "spotify_id": r.get("spotify_id"),
            }
            for r in results
        ]
    })


if __name__ == "__main__":
    app.run(debug=True)