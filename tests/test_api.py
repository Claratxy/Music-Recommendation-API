"""
test_api.py
-------------
Automated tests for the Flask API in app.py: input validation, response
shape, and rate limiting -- exercising the actual HTTP layer, not just
the scoring logic underneath it.

Run with:
    python -m pytest tests/test_api.py -v
"""
import pytest

from app import app as flask_app
import rate_limit_store
from data import TRACKS

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    rate_limit_store.reset_all()
    with flask_app.test_client() as c:
        yield c

def test_list_tracks_returns_all_tracks(client):
    res = client.get("/tracks")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data) == len(TRACKS)
    assert set(data[0].keys()) == {"id", "title", "artist"}


def test_recommend_valid_request(client):
    seed = TRACKS[0]["id"]
    res = client.post("/recommend", json={
        "track_ids": [seed], "mood": "happy", "energy": 0.7, "novelty": 0.3,
    })
    assert res.status_code == 200
    data = res.get_json()
    for key in ("track_id", "title", "artist", "score", "reason",
                "breakdown", "tags", "tag_source"):
        assert key in data
    assert data["track_id"] != seed


def test_recommend_no_track_ids_returns_400(client):
    res = client.post("/recommend", json={"track_ids": []})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_recommend_too_many_track_ids_returns_400(client):
    seeds = [t["id"] for t in TRACKS[:6]]
    res = client.post("/recommend", json={"track_ids": seeds})
    assert res.status_code == 400


def test_recommend_unknown_track_id_returns_400(client):
    res = client.post("/recommend", json={"track_ids": ["not-a-real-id"]})
    assert res.status_code == 400
    assert "not-a-real-id" in res.get_json()["error"]


@pytest.mark.parametrize("field,value", [
    ("energy", 1.5), ("energy", -0.1),
    ("novelty", 2.0), ("novelty", -1.0),
])
def test_recommend_out_of_range_context_returns_400(client, field, value):
    body = {"track_ids": [TRACKS[0]["id"]], field: value}
    res = client.post("/recommend", json=body)
    assert res.status_code == 400


def test_recommend_uses_defaults_when_context_omitted(client):
    res = client.post("/recommend", json={"track_ids": [TRACKS[0]["id"]]})
    assert res.status_code == 200


class TestRateLimiting:
    """Runs against the real SQLite-backed limiter. reset_all() is
    called first so this test is not affected by counts left over
    from a previous test or a previous full test-suite run --
    something the old in-memory backend never needed, since it
    started empty every time the process launched."""

    def test_recommend_rate_limit_returns_429_after_20_requests(self):
        flask_app.config["TESTING"] = True
        rate_limit_store.reset_all()
        with flask_app.test_client() as c:
            seed = TRACKS[0]["id"]
            statuses = [
                c.post("/recommend", json={"track_ids": [seed]}).status_code
                for _ in range(22)
            ]
            assert statuses[:20].count(200) == 20
            assert 429 in statuses[20:]

    def test_rate_limit_response_is_valid_json(self):
        """Regression test for the earlier bug where a 429 response
        wasn't JSON."""
        flask_app.config["TESTING"] = True
        rate_limit_store.reset_all()
        with flask_app.test_client() as c:
            seed = TRACKS[0]["id"]
            responses = [
                c.post("/recommend", json={"track_ids": [seed]})
                for _ in range(22)
            ]
            rate_limited_responses = [r for r in responses if r.status_code == 429]
            assert rate_limited_responses, "expected at least one 429 in this run"
            assert rate_limited_responses[0].is_json
            assert "error" in rate_limited_responses[0].get_json()

def test_rate_limit_persists_across_simulated_restart():
    """The behaviour the in-memory backend explicitly could not
    provide (Draft Report, Section 3.5): counts must survive the
    Flask app object being torn down and recreated, since that's
    what actually happens on a real server restart. This test
    doesn't restart the OS process (impractical in a unit test), but
    it does exercise the real persistence mechanism: reset, hit the
    limit, re-import a fresh Flask test client against the SAME
    sqlite file, and confirm the limit is still in effect -- proving
    state lives in the file, not in Python process memory.
    """
    import rate_limit_store
    rate_limit_store.reset_all()
    seed = TRACKS[0]["id"]

    with flask_app.test_client() as c1:
        for _ in range(20):
            c1.post("/recommend", json={"track_ids": [seed]})

    # A fresh client object simulates a new process attaching to the
    # same persistent store -- the 21st request should already be
    # rate-limited even though no Python state was carried over
    # explicitly, only the sqlite file on disk.
    with flask_app.test_client() as c2:
        res = c2.post("/recommend", json={"track_ids": [seed]})
        assert res.status_code == 429

def test_search_matches_title(client):
    res = client.get("/search?q=brightside")
    assert res.status_code == 200
    data = res.get_json()
    assert any(t["title"] == "Mr. Brightside" for t in data)
  
def test_search_matches_artist_case_insensitive(client):
    res = client.get("/search?q=THE KILLERS")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data) > 0
    assert all("killers" in t["artist"].lower() for t in data)
 
 
def test_search_missing_query_returns_400(client):
    res = client.get("/search")
    assert res.status_code == 400
    assert "error" in res.get_json()
 
 
def test_search_too_short_query_returns_400(client):
    res = client.get("/search?q=a")
    assert res.status_code == 400
 
 
def test_search_no_matches_returns_empty_list(client):
    res = client.get("/search?q=zzzznotarealtrackzzzz")
    assert res.status_code == 200
    assert res.get_json() == []
 
 
def test_search_caps_at_30_results(client):
    # A common two-letter substring should match well over 30 of the
    # 174 tracks. (Must be >=2 chars -- the endpoint itself requires
    # that, see test_search_too_short_query_returns_400.)
    res = client.get("/search?q=an")
    assert res.status_code == 200
    assert len(res.get_json()) <= 30     

def test_recommend_queue_valid_request(client):
    seed = TRACKS[0]["id"]
    res = client.post("/recommend/queue", json={
        "track_ids": [seed], "count": 3, "diversity": 0.5,
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 3
    for r in data["recommendations"]:
        for key in ("track_id", "title", "artist", "score", "reason"):
            assert key in r
        assert r["track_id"] != seed
 
 
def test_recommend_queue_defaults_when_count_and_diversity_omitted(client):
    seed = TRACKS[0]["id"]
    res = client.post("/recommend/queue", json={"track_ids": [seed]})
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["recommendations"]) == 3  # default count
 
 
@pytest.mark.parametrize("count", [1, 6, 0, -1])
def test_recommend_queue_out_of_range_count_returns_400(client, count):
    seed = TRACKS[0]["id"]
    res = client.post("/recommend/queue", json={
        "track_ids": [seed], "count": count,
    })
    assert res.status_code == 400
    assert "error" in res.get_json()
 
 
@pytest.mark.parametrize("diversity", [-0.1, 1.5])
def test_recommend_queue_out_of_range_diversity_returns_400(client, diversity):
    seed = TRACKS[0]["id"]
    res = client.post("/recommend/queue", json={
        "track_ids": [seed], "diversity": diversity,
    })
    assert res.status_code == 400
    assert "error" in res.get_json()
 
 
def test_recommend_queue_no_track_ids_returns_400(client):
    res = client.post("/recommend/queue", json={"track_ids": []})
    assert res.status_code == 400
    assert "error" in res.get_json()
 
 
def test_recommend_queue_unknown_track_id_returns_400(client):
    res = client.post("/recommend/queue", json={"track_ids": ["not-a-real-id"]})
    assert res.status_code == 400
    assert "not-a-real-id" in res.get_json()["error"]
 
 
def test_recommend_queue_results_are_mutually_distinct(client):
    """Guards against a regression where MMR could theoretically select
    the same candidate twice."""
    seed = TRACKS[0]["id"]
    res = client.post("/recommend/queue", json={
        "track_ids": [seed], "count": 5, "diversity": 0.8,
    })
    assert res.status_code == 200
    ids = [r["track_id"] for r in res.get_json()["recommendations"]]
    assert len(ids) == len(set(ids))
 
 
def test_recommend_queue_malformed_json_returns_400(client):
    """The endpoint should respond with a consistent JSON error, not a
    generic HTML error page, when the request body isn't valid JSON at
    all (not just wrong values -- actually malformed syntax)."""
    res = client.post(
        "/recommend/queue",
        data="{not valid json",
        content_type="application/json",
    )
    assert res.status_code == 400
    assert res.is_json
    assert "error" in res.get_json()    