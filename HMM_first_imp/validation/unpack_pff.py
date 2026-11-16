"""Unpack the PFF FC World Cup 2022 download (Google Drive splits it into several zips) into one tidy folder:

    data/pff/raw/tracking/<game>.jsonl.bz2
    data/pff/raw/metadata/<game>.json
    data/pff/raw/rosters/<game>.json
    data/pff/raw/events/...            (event data: one events.json or one file per game)
    data/pff/raw/other/...             (anything else, original path kept)

From the repository root (the zips are only read, never changed or deleted):

    python HMM_first_imp/validation/unpack_pff.py C:/Users/giuse/Downloads/drive-download-20261109T160142Z-1-00*.zip
    python HMM_first_imp/validation/unpack_pff.py --dry-run <zips...>       # only list what would go where

A file already there with the same size is skipped, so the command can be run again after an interruption.
data/ is gitignored: nothing of this reaches git. Then: validation/fetch_pff.py --inspect.
"""
import argparse
import collections
import glob
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def kind_of(name, head):
    """'tracking' | 'metadata' | 'rosters' | 'events' | 'other' from the member path and its first bytes."""
    low = name.replace("\\", "/").lower()
    base = os.path.basename(low)
    stem = base.split(".")[0]
    if base.endswith(".jsonl.bz2") or base.endswith(".jsonl"):
        return "tracking"
    if not base.endswith(".json"):
        return "other"
    folders = low.split("/")[:-1]
    if any("event" in f for f in folders) or "event" in base:
        return "events"
    if any("roster" in f for f in folders):
        return "rosters"
    if any("meta" in f for f in folders):
        return "metadata"
    if any(k in head for k in ('"gameEvents"', '"possessionEvents"', '"game_event"')):
        return "events"
    if stem.isdigit() and '"homeTeam"' in head and '"fps"' in head:
        return "metadata"
    if stem.isdigit() and '"shirtNumber"' in head and '"positionGroupType"' in head:
        return "rosters"
    return "other"


def target_of(kind, name):
    base = os.path.basename(name.replace("\\", "/"))
    if kind == "other":
        return os.path.join("other", *[p for p in name.replace("\\", "/").split("/") if p not in ("", "..", ".")])
    return os.path.join(kind, base)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zips", nargs="+", help="the downloaded zip files (wildcards allowed)")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "pff", "raw"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = sorted({p for z in args.zips for p in (glob.glob(z) or [z])})
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print("not found:", missing)
        return 1
    counts, sizes, clashes = collections.Counter(), collections.Counter(), []
    seen = {}
    for zp in paths:
        print("==", zp, flush=True)
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                head = ""
                if info.filename.lower().endswith(".json"):
                    with z.open(info) as h:
                        head = h.read(4096).decode("utf-8", "replace")
                kind = kind_of(info.filename, head)
                rel = target_of(kind, info.filename)
                if rel in seen and seen[rel] != (zp, info.filename):
                    if seen[rel][2] != info.file_size:
                        clashes.append((rel, seen[rel][1], info.filename))
                    continue
                seen[rel] = (zp, info.filename, info.file_size)
                counts[kind] += 1
                sizes[kind] += info.file_size
                dest = os.path.join(args.out, rel)
                if args.dry_run:
                    if counts[kind] <= 3:
                        print("  {:9s} {}  ->  {}".format(kind, info.filename, rel))
                    continue
                if os.path.isfile(dest) and os.path.getsize(dest) == info.file_size:
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + ".part"
                with z.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(tmp, dest)
                if counts[kind] % 10 == 0:
                    print("  {} {} files".format(kind, counts[kind]), flush=True)
    print("\n{} into {}".format("would go" if args.dry_run else "unpacked", args.out))
    for k in ("tracking", "metadata", "rosters", "events", "other"):
        print("  {:9s} {:4d} files  {:8.0f} MB".format(k, counts[k], sizes[k] / 2 ** 20))
    if clashes:
        print("same name, different size (first kept):", clashes[:10])
    tr = {os.path.basename(r).split(".")[0] for r in seen if r.startswith("tracking")}
    me = {os.path.basename(r).split(".")[0] for r in seen if r.startswith("metadata")}
    print("games with tracking: {}, with metadata too: {}".format(len(tr), len(tr & me)))
    if tr - me:
        print("tracking without metadata:", sorted(tr - me)[:20])
    others = sorted(r for r in seen if r.startswith("other"))
    if others:
        print("other files (first 15):", others[:15])
    print("\nnext: python HMM_first_imp/validation/fetch_pff.py --inspect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
