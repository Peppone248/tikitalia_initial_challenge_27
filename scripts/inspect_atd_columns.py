"""Show the structure of the ATD: raw file layout, metadata, and every column kloppy produces.

    python scripts/inspect_atd_columns.py                 # tactical, 1 frame per second (fast)
    python scripts/inspect_atd_columns.py --full          # every frame (minutes, several GB RAM)
    python scripts/inspect_atd_columns.py --source broadcast
    python scripts/inspect_atd_columns.py --out cache/atd_columns.txt

Prints structure only: field counts, column names, dtypes, missing-value rates and the distinct
values of the few categorical columns (ball state, owning team). No coordinates, no rows.
"""
import argparse
import io
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import DEFAULT_SOURCE, GAMES, GAME, require  # noqa: E402

SLOT_COL = re.compile(r"^(?P<slot>.+?)_(?P<field>[a-z]+)$")
CONTEXT_HINTS = ("ball_state", "ball_owning_team", "period", "timestamp", "ball_z")


def h(out, title):
    out.write("\n" + "=" * 78 + "\n{}\n".format(title) + "=" * 78 + "\n")


def raw_structure(raw_path, out, n_lines=2000):
    """Field layout of the raw .txt, from the first lines. Counts only, no values."""
    h(out, "1. RAW FILE  {}".format(os.path.basename(raw_path)))
    parts_count, groups_count, player_fields, ball_fields = Counter(), Counter(), Counter(), Counter()
    with open(raw_path) as handle:
        for i, line in enumerate(handle):
            if i >= n_lines:
                break
            parts = line.rstrip("\n").split(":")
            parts_count[len(parts)] += 1
            if len(parts) < 2:
                continue
            groups = parts[1].split(";")
            groups_count[len(groups)] += 1
            for g in groups:
                player_fields[len(g.split(","))] += 1
            if len(parts) >= 3:
                ball_fields[len(parts[2].split(","))] += 1
    out.write("lines inspected: {}\n".format(min(i + 1, n_lines)))
    out.write("':'-separated parts per line: {}   (expected 3: frame_id : players : ball)\n".format(dict(parts_count)))
    out.write("';'-separated player groups per line: {}\n".format(dict(groups_count)))
    out.write("','-separated fields per player group: {}\n".format(dict(player_fields)))
    out.write("','-separated fields in the ball group: {}\n".format(dict(ball_fields)))
    n_groups = groups_count.most_common(1)[0][0] if groups_count else 0
    pf = player_fields.most_common(1)[0][0] if player_fields else 0
    bf = ball_fields.most_common(1)[0][0] if ball_fields else 0
    out.write("=> values per line: 1 (frame_id) + {} x {} (tracks) + {} (ball) = {}\n".format(
        n_groups, pf, bf, 1 + n_groups * pf + bf))


def metadata_structure(meta_path, out):
    """Element tree of the metadata XML (tags, counts, attribute names) and the field spec."""
    h(out, "2. METADATA  {}".format(os.path.basename(meta_path)))
    root = ET.parse(meta_path).getroot()

    def strip(tag):
        return tag.split("}", 1)[-1]

    tags = OrderedDict()
    for el in root.iter():
        t = strip(el.tag)
        entry = tags.setdefault(t, [0, set()])
        entry[0] += 1
        entry[1].update(el.attrib.keys())
    out.write("element tags (count, attribute names):\n")
    for t, (c, attrs) in tags.items():
        out.write("  {:<28} {:>5}  {}\n".format(t, c, ", ".join(sorted(attrs)) or "-"))

    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in root.iter() if strip(t.tag) == "Team"]
    out.write("\nteams (first = home): {}\n".format(teams))
    per_team = Counter(p.get("teamId") for p in root.iter() if strip(p.tag) == "Player")
    out.write("track slots per team id: {}\n".format(dict(per_team)))
    for name in ("FrameRate", "FieldSize", "Length", "Width"):
        for el in root.iter():
            if strip(el.tag) == name and (el.text or "").strip():
                out.write("{}: {}\n".format(name, el.text.strip()))
                break

    # The data spec says, in order, what each field of a raw line is. Summarised, not dumped.
    specs = [el for el in root.iter() if strip(el.tag) == "DataFormatSpecification"]
    out.write("\nDataFormatSpecification blocks: {}  (frame ranges: {})\n".format(
        len(specs), ", ".join("{}-{}".format(s.get("startFrame"), s.get("endFrame")) for s in specs[:6])
        + (" ..." if len(specs) > 6 else "")))
    if specs:
        first = specs[0]
        refs = [el for el in first.iter() if strip(el.tag) in ("PlayerChannelRef", "BallChannelRef")]
        player_refs = [el.get("playerChannelId") or "" for el in refs if strip(el.tag) == "PlayerChannelRef"]
        ball_refs = [el.get("channelId") or "" for el in refs if strip(el.tag) == "BallChannelRef"]
        kinds = OrderedDict()
        for r in player_refs:
            kinds[r.rsplit("_", 1)[-1] if "_" in r else r] = None
        out.write("  first block: {} player channel refs -> fields per player: {}\n".format(
            len(player_refs), list(kinds)))
        out.write("  ball channels: {}\n".format(ball_refs))
        frame_reg = [el.get("name") for el in first.iter() if strip(el.tag) == "StringRegister"]
        out.write("  string registers: {}\n".format(frame_reg))
    channels = [el for el in root.iter() if strip(el.tag) in ("Channel", "PlayerChannel")]
    if channels:
        out.write("\nchannels (one per value type):\n")
        seen = set()
        for el in channels:
            key = (strip(el.tag), el.get("channelId") or el.get("id"))
            if key in seen:
                continue
            seen.add(key)
            name = el.findtext("{*}Name") or el.findtext("Name") or ""
            unit = el.findtext("{*}Unit") or el.findtext("Unit") or ""
            out.write("  {:<14} id={:<22} name={:<20} unit={}\n".format(key[0], str(key[1]), name.strip(), unit.strip()))
            if len(seen) >= 30:
                out.write("  ...\n")
                break


def kloppy_columns(meta_path, raw_path, out, full):
    h(out, "3. KLOPPY  dataset.to_df(){}".format("" if full else "  (sample: 1 frame per second)"))
    try:
        from kloppy import metrica
    except ImportError:
        out.write("kloppy not installed: pip install -r requirements-starter.txt\n")
        return
    kwargs = {} if full else {"sample_rate": 1.0 / GAMES[GAME]["fps"]}
    print("loading with kloppy ...", file=sys.stderr)
    ds = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path, **kwargs)
    df = ds.to_df()
    md = ds.metadata

    out.write("rows: {}   columns: {}\n".format(len(df), len(df.columns)))
    out.write("pitch: {}   frame_rate: {}   orientation: {}\n".format(
        getattr(md, "pitch_dimensions", None), getattr(md, "frame_rate", None), getattr(md, "orientation", None)))
    out.write("teams: {}\n".format([(t.team_id, t.name, t.ground) for t in md.teams]))

    slot_cols, other_cols = OrderedDict(), []
    for c in df.columns:
        m = SLOT_COL.match(str(c))
        if m and m.group("slot") not in ("ball",) and not str(c).startswith(CONTEXT_HINTS):
            slot_cols.setdefault(m.group("slot"), []).append(m.group("field"))
        else:
            other_cols.append(c)

    out.write("\n-- non-track columns ({}) --\n".format(len(other_cols)))
    out.write("  {:<28} {:<16} {:>9}  {}\n".format("column", "dtype", "missing", "distinct values (categorical only)"))
    for c in other_cols:
        s = df[c]
        miss = 100.0 * s.isna().mean()
        extra = ""
        if s.dtype == object or str(s.dtype).startswith("category") or str(c).startswith(CONTEXT_HINTS[:2]):
            vals = s.dropna().astype(str).value_counts()
            extra = ", ".join("{} ({:.1f}%)".format(v, 100.0 * k / len(s)) for v, k in vals.head(8).items())
            if len(vals) > 8:
                extra += ", ... ({} distinct)".format(len(vals))
        out.write("  {:<28} {:<16} {:>8.1f}%  {}\n".format(str(c), str(s.dtype), miss, extra))

    out.write("\n-- track columns: {} slots x fields {} = {} columns --\n".format(
        len(slot_cols), sorted({tuple(v) for v in slot_cols.values()}), sum(len(v) for v in slot_cols.values())))
    out.write("  {:<10} {:<20} {:>9}\n".format("slot", "fields", "missing"))
    for slot, fields in slot_cols.items():
        cols = ["{}_{}".format(slot, f) for f in fields]
        miss = 100.0 * df[cols[0]].isna().mean()
        out.write("  {:<10} {:<20} {:>8.1f}%\n".format(str(slot), ",".join(fields), miss))

    out.write("\n-- all column names, in order --\n")
    names = [str(c) for c in df.columns]
    for i in range(0, len(names), 6):
        out.write("  " + "  ".join("{:<14}".format(n) for n in names[i:i + 6]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=("tactical", "broadcast"))
    ap.add_argument("--full", action="store_true", help="load every frame (slow, memory-heavy)")
    ap.add_argument("--out", help="also write the report to this file")
    ap.add_argument("--meta", help=argparse.SUPPRESS)  # testing on another EPTS pair
    ap.add_argument("--raw", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.meta and args.raw:
        meta_path, raw_path = args.meta, args.raw
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], source=args.source)

    out = io.StringIO()
    out.write("ATD structure - source: {}\n".format(args.source))
    raw_structure(raw_path, out)
    metadata_structure(meta_path, out)
    kloppy_columns(meta_path, raw_path, out, args.full)

    text = out.getvalue()
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
