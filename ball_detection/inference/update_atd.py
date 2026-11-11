"""
Pipeline step 4: creates a copy of the ATD (Metrica, FIFA/EPTS format) in which the ball is
replaced by the ensemble's one (ball.csv). The original ATD is not modified.

For every frame of the video:
  - if ball.csv has the ball (confirmed / tracked / recovered / interpolated), the ball's x and y
    in the ATD become the ensemble's, normalised 0-1 as in the ATD
    (pitch_x_m / 105.3, pitch_y_m / 68.0); the speed s becomes the one computed on the ensemble's
    track (m/s, central difference), or NaN if a neighbouring frame has no ball;
  - if ball.csv has no ball (none), the ATD ball stays the original one.
Players and everything else in the row stay identical. The metadata XML is copied next to the
new .txt, so the pair can be read with kloppy like the original.

Indices: the ATD row with frameCount = k corresponds to frame k - 1 of the video.

Output (default, next to the original ATD):
  BAR-ATH_tactical_atd_tikitalia.txt / .xml

Usage (from the project root):
  python -m inference.update_atd
  python -m inference.update_atd --atd <path>/BAR-ATH_tactical_atd.txt --suffix tikitalia
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import ALL_MODELS, DEFAULT_MODELS, CHALLENGE_DATA, DEFAULT_VIDEO, ball_path, set_video
from utils.pitch import PITCH_M


def ball_speed(df, fps):
    """Ball speed in m/s with a central difference (NaN if a neighbour is missing)."""
    p = df[["pitch_x_m", "pitch_y_m"]].to_numpy(dtype=float)
    s = np.full(len(p), np.nan)
    d = np.linalg.norm(p[2:] - p[:-2], axis=1) * fps / 2
    s[1:-1] = d
    return s


def main():
    p = argparse.ArgumentParser(description="ATD with the ensemble's ball")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--atd", type=Path, default=CHALLENGE_DATA / "BAR-ATH_tactical_atd.txt")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    p.add_argument("--suffix", default="tikitalia", help="new file: <atd>_<suffix>.txt/.xml")
    p.add_argument("--fps", type=float, default=25)
    args = p.parse_args()

    set_video(args.video)
    src = ball_path([m for m in ALL_MODELS if m in args.models])
    ball = pd.read_csv(src).set_index("frame").sort_index()
    ball["speed"] = ball_speed(ball, args.fps)
    has = ball["status"] != "none"
    bx = (ball["pitch_x_m"] / PITCH_M[0]).to_numpy()
    by = (ball["pitch_y_m"] / PITCH_M[1]).to_numpy()
    bs = ball["speed"].to_numpy()
    # frame -> row of ball.csv, for the frames where the ensemble has the ball. ball.csv may cover
    # only part of the video (e.g. a test with --start/--frames), so rows are looked up by frame.
    rows = {int(f): i for i, f in enumerate(ball.index) if has.iloc[i]}

    out_txt = args.atd.with_name(f"{args.atd.stem}_{args.suffix}.txt")
    out_xml = out_txt.with_suffix(".xml")
    if out_txt.resolve() == args.atd.resolve():
        raise SystemExit("[ERROR] the new file is the same as the original ATD")

    n = replaced = added = kept = 0
    with open(args.atd, encoding="utf-8") as fin, open(out_txt, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.rstrip("\n")
            head, old_ball = line.rsplit(":", 1)
            frame = int(head.split(":", 1)[0]) - 1          # frameCount 1 = frame 0 of the video
            n += 1
            if frame in rows:
                i = rows[frame]
                s = "NaN" if np.isnan(bs[i]) else f"{bs[i]}"
                new_ball = f"{bx[i]:.5f},{by[i]:.5f},{s}"
                if old_ball.startswith("NaN"):
                    added += 1
                else:
                    replaced += 1
                line = f"{head}:{new_ball}"
            elif not old_ball.startswith("NaN"):
                kept += 1
            fout.write(line + "\n")
    shutil.copyfile(args.atd.with_suffix(".xml"), out_xml)

    tot = replaced + added + kept
    print(f"[INFO] {n} rows; ball from the ensemble ({src.parent.name}): {replaced + added} "
          f"({replaced} replaced, {added} added where the ATD had none); "
          f"original ATD ball kept where the ensemble has no ball: {kept}")
    print(f"[INFO] frames with the ball: original ATD {replaced + kept} ({100 * (replaced + kept) / n:.1f}%), "
          f"new ATD {tot} ({100 * tot / n:.1f}%)")
    print(f"[INFO] Saved {out_txt} and {out_xml}")


if __name__ == "__main__":
    main()
