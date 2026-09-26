"""
append_low_energy.py
----------------------
Merges LOW_ENERGY_ADDITIONS (from rebalance_catalogue.py's output,
low_energy_additions.py) into the existing track_catalog.py, the same
way merge_catalog.py originally built it -- so the output stays a
drop-in file with the same three exported names (TRACK_CATALOG,
FALLBACK_TAGS, TRANSITION_COUNTS) that data.py and scorer.py expect.

This exists so don't hand-edit the auto-generated track_catalog.py
directly (easy to break the Python syntax in a 179-entry list by hand).

Usage (run in the project root, where track_catalog.py and
low_energy_additions.py both already exist):
    python append_low_energy.py
"""

import importlib.util
import json
import re
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _normalize_title(title):
    return re.sub(r"\s*[-(].*$", "", title).strip().lower()


def main():
    current = _load_module(os.path.join(_ROOT, "track_catalog.py"), "current_catalog")
    additions = _load_module(os.path.join(_HERE, "low_energy_additions.py"), "low_energy_additions")

    existing_catalog = list(current.TRACK_CATALOG)
    existing_fallback = dict(current.FALLBACK_TAGS)
    existing_transitions = dict(current.TRANSITION_COUNTS)

    existing_keys = {
        _normalize_title(t["title"]) + "::" + t["artist"].lower()
        for t in existing_catalog
    }

    added, skipped = 0, 0
    for track in additions.LOW_ENERGY_ADDITIONS:
        key = _normalize_title(track["title"]) + "::" + track["artist"].lower()
        if key in existing_keys:
            print(f"  [skip] {track['id']} {track['artist']} - {track['title']}: "
                  f"already in catalogue (same normalized title/artist)")
            skipped += 1
            continue
        existing_catalog.append(track)
        existing_keys.add(key)
        # Placeholder fallback tag, same pattern merge_catalog.py used for
        # newly-sampled tracks -- fetch_tags.py will overwrite this with
        # real MusicBrainz tags on the next run.
        existing_fallback[track["id"]] = [track.get("genre_hint", "unknown")]
        added += 1

    def py_repr(v):
        if v is None:
            return "None"
        if isinstance(v, bool):
            return repr(v)
        return json.dumps(v)

    with open(os.path.join(_ROOT, "track_catalog.py"), "w", encoding="utf-8") as out:
        out.write('"""\n')
        out.write("track_catalog.py\n")
        out.write("-----------------\n")
        out.write(f"Catalogue: {len(existing_catalog)} tracks. Includes "
                   f"{added} low-energy additions appended by "
                   f"append_low_energy.py to fix the energy skew diagnosed "
                   f"in the Draft Report (Section 5.6).\n")
        out.write('"""\n\n')

        out.write("TRACK_CATALOG = [\n")
        for t in existing_catalog:
            fields = ", ".join(f'"{k}": {py_repr(v)}' for k, v in t.items())
            out.write(f"    {{{fields}}},\n")
        out.write("]\n\n")

        out.write("FALLBACK_TAGS = {\n")
        for tid, tags in existing_fallback.items():
            out.write(f"    {json.dumps(tid)}: {json.dumps(tags)},\n")
        out.write("}\n\n")

        out.write("TRANSITION_COUNTS = {\n")
        for (a, b), count in existing_transitions.items():
            out.write(f"    ({json.dumps(a)}, {json.dumps(b)}): {count},\n")
        out.write("}\n")

    print(f"\nAdded {added} new low-energy tracks "
          f"({skipped} skipped as likely duplicates).")
    print(f"track_catalog.py rewritten: {len(existing_catalog)} tracks total.")
    print("\nNote: the new tracks have NO real transitions yet (they weren't")
    print("in the original 12.9M-row playlist mining run), and NO real")
    print("MusicBrainz tags or Spotify IDs yet -- next steps below.")


if __name__ == "__main__":
    main()
