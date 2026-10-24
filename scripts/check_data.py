"""Check that the match data is where games.py expects it.

    python scripts/check_data.py

Exits 1 if any tactical asset is missing (broadcast is optional).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import DATA_DIR, asset_paths  # noqa: E402


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "{:.1f} {}".format(n, unit) if unit != "B" else "{} B".format(n)
        n /= 1024.0


def main():
    print("data dir: {}{}".format(DATA_DIR, "  (from MTAH_DATA_DIR)" if os.environ.get("MTAH_DATA_DIR") else ""))
    missing_required = False
    for source in ("tactical", "broadcast"):
        print("\n[{}]{}".format(source, "" if source == "tactical" else "  optional"))
        for key, path in asset_paths(source=source).items():
            ok = os.path.isfile(path)
            size = human(os.path.getsize(path)) if ok else "MISSING"
            print("  {:<4} {:<14} {:>10}  {}".format("ok" if ok else "!!", key, size, os.path.relpath(path, ROOT)))
            if not ok and source == "tactical":
                missing_required = True
    if missing_required:
        print("\nTactical assets missing - see TEAM.md section 2.")
        return 1
    print("\nAll tactical assets present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
