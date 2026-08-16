"""
test_prototype.py
-------------------
Runs a small set of test cases against the prototype and prints the
results. 

Run with:
    python test_prototype.py
"""

from scorer import recommend_next_track

TEST_CASES = [
    {
        "name": "Case 1: low novelty, energetic mood, indie-rock seed",
        "track_ids": ["t01", "t02"],
        "mood": "energetic",
        "energy": 0.8,
        "novelty": 0.1,
    },
    {
        "name": "Case 2: same seed, high novelty",
        "track_ids": ["t01", "t02"],
        "mood": "energetic",
        "energy": 0.8,
        "novelty": 0.9,
    },
    {
        "name": "Case 3: calm mood, acoustic seed",
        "track_ids": ["t05", "t06"],
        "mood": "calm",
        "energy": 0.3,
        "novelty": 0.2,
    },
    {
        "name": "Case 4: cold-start, single seed track, hip-hop",
        "track_ids": ["t07"],
        "mood": "neutral",
        "energy": 0.6,
        "novelty": 0.4,
    },
    {
        "name": "Case 5: electronic seed, strong discovery request",
        "track_ids": ["t03", "t04"],
        "mood": "happy",
        "energy": 0.85,
        "novelty": 0.8,
    },
]

if __name__ == "__main__":
    for case in TEST_CASES:
        print("=" * 70)
        print(case["name"])
        print(f"Input: track_ids={case['track_ids']}, mood={case['mood']}, "
              f"energy={case['energy']}, novelty={case['novelty']}")
        result = recommend_next_track(
            case["track_ids"], mood=case["mood"],
            energy=case["energy"], novelty=case["novelty"], top_n=1
        )[0]
        print(f"Output: track_id={result['track_id']} "
              f"({result['title']} - {result['artist']})")
        print(f"        score={result['confidence_score']}")
        print(f"        reason: {result['reason']}")
        print()
