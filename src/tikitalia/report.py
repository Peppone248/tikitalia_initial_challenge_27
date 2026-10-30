"""Per-rule report. The printed summary is AGGREGATED ONLY (counts, shares, quantiles): it carries
no frame numbers and no positions, so it can be discussed outside the team machines. Episode
lists with frame numbers go to CSV under outputs/ (gitignored) for local debugging only.
"""

import os

import numpy as np

STRENGTH_BINS = (0.0, 0.25, 0.5, 0.75, 1.0001)


def _q(values, qs=(0.1, 0.5, 0.9)):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return "n/a"
    return " / ".join("{:.2f}".format(np.quantile(values, q)) for q in qs)


def _share(n, total):
    return "{} ({:.1f}%)".format(n, 100.0 * n / total) if total else "0"


def _bin_label(lo, hi):
    return "[{:.2f}, {:.2f})".format(lo, min(hi, 1.0))


def general(match, context):
    graded = match.graded
    lines = ["== General ==",
             "frames {}  graded {}".format(match.n, int(graded.sum())),
             "ball visible on graded frames: {:.1f}%".format(
                 100.0 * match.ball_visible[graded].mean()),
             "ball jumps removed: {}".format(match.diagnostics.get("ball_jumps_removed")),
             "SIGMA_POS (ball jitter, estimated): {:.3f} m".format(
                 match.diagnostics.get("sigma_pos_m"))]
    visible_graded = graded & match.ball_visible
    if visible_graded.any():
        lines.append("ball 'still' (speed < V_BALL_STILL) on visible graded frames: {:.1f}%  "
                     "| speed p10/p50/p90 m/s: {}".format(
                         100.0 * match.diagnostics.get("still_share", np.nan),
                         _q(match.ball_speed[visible_graded & ~np.isnan(match.ball_speed)])))
    kos = context["kickoffs"]
    lines.append("kick-offs detected: {}  (first of half: {}; after: {}; kicker H/A {}/{}; "
                 "ball seen at centre {})".format(
                     len(kos), sum(k["first_of_half"] for k in kos),
                     sum(not k["first_of_half"] for k in kos),
                     sum(k["kicker"] == 0 for k in kos), sum(k["kicker"] == 1 for k in kos),
                     sum(k.get("ball_seen", False) for k in kos)))
    for half, info in sorted(context["attack_method"].items()):
        lines.append("attack direction half {}: {}".format(half, info))
    lines.append("attack direction consistent (H opposite A, flips at halftime): {}".format(
        _attack_dir_ok(context["attack_dir"])))
    return lines


def _attack_dir_ok(attack_dir):
    try:
        return (attack_dir[(0, 1)] == -attack_dir[(1, 1)]
                and attack_dir[(0, 2)] == -attack_dir[(1, 2)]
                and attack_dir[(0, 1)] == -attack_dir[(0, 2)])
    except KeyError:
        return "incomplete"


def ball_out(output, match):
    eps = [e for e in output.episodes if e.get("graded")]
    graded = match.graded
    labelled = int((output.fired & graded).sum())
    lines = ["== D-OUT ==",
             "episodes on graded frames: {}  (all frames: {})".format(len(eps), len(output.episodes)),
             "graded frames labelled D: {}".format(_share(labelled, int(graded.sum())))]
    if not eps:
        return lines
    for line in ("touch", "goal"):
        lines.append("  crossed {:<5} line: {}".format(line, sum(e["line"] == line for e in eps)))
    lines.append("duration s (p10 / p50 / p90): {}".format(_q([e["duration_s"] for e in eps])))
    lines.append("peak distance beyond line m (p10 / p50 / p90): {}".format(
        _q([e["peak_dist_m"] for e in eps])))
    lines.append("peak z (p10 / p50 / p90): {}".format(_q([e["peak_z"] for e in eps])))
    lines.append("episodes ending with the ball lost: {}".format(
        _share(sum(e["end_reason"] == "ball_lost" for e in eps), len(eps))))
    lines.append("time until the ball is seen inside again, s (p10 / p50 / p90): {}".format(
        _q([e["reappear_inside_s"] for e in eps])))

    lines.append("Smart Tagging coherence (restart type vs line crossed, within ST_MATCH_WIN):")
    for line in ("touch", "goal"):
        sub = [e for e in eps if e["line"] == line]
        counts = {c: sum(e["st_coherence"] == c for e in sub)
                  for c in ("coherent", "incoherent", "no_tag")}
        lines.append("  {:<5} n={:<4} coherent {}  incoherent {}  no tag {}".format(
            line, len(sub), counts["coherent"], counts["incoherent"], counts["no_tag"]))

    lines.append("strength bins -> n | share coherent among tagged | median duration s | "
                 "share ending ball lost")
    for lo, hi in zip(STRENGTH_BINS[:-1], STRENGTH_BINS[1:]):
        sub = [e for e in eps if lo <= e["strength"] < hi]
        tagged = [e for e in sub if e["st_coherence"] != "no_tag"]
        coherent = sum(e["st_coherence"] == "coherent" for e in tagged)
        lines.append("  {}  n={:<4} {:<14} {:<6} {}".format(
            _bin_label(lo, hi), len(sub),
            "{}/{}".format(coherent, len(tagged)) if tagged else "-",
            _q([e["duration_s"] for e in sub], (0.5,)) if sub else "-",
            _share(sum(e["end_reason"] == "ball_lost" for e in sub), len(sub)) if sub else "-"))
    return lines


def st_restart_recall(output, match, window_s):
    """How many Smart Tagging out-of-play restarts have a D-OUT episode just before them."""
    starts = np.array([e["start"] for e in output.episodes]) if output.episodes else np.array([])
    lines = ["Smart Tagging restarts preceded by a D-OUT episode (within {:.0f} s):".format(window_s)]
    for kind in ("throw-in", "corner kick", "goal kick"):
        tags = [t for t in match.smart_tagging if t["type"] == kind]
        hit = 0
        for t in tags:
            lo = t["start_frame"] - window_s * match.fps
            hit += bool(starts.size and np.any((starts >= lo) & (starts < t["end_frame"])))
        lines.append("  {:<12} {}/{}".format(kind, hit, len(tags)))
    return lines


def goal(output):
    eps = output.episodes
    goals = [e for e in eps if e.get("status") in ("confirmed", "no_candidate")]
    lines = ["== D-GOAL (from the kick-off backwards) ==",
             "kick-offs after a goal analysed: {}".format(len(goals))]
    by_source = {}
    for e in goals:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
    lines.append("  goal found by source: {}".format(by_source))
    lines.append("goals -> half | source | z | strength | ST GOALS | ST KO after goal | "
                 "s goal->kick-off | scorer | kicker s at spot | ball seen at kick-off | "
                 "ball after kick-off m from centre | kick-off rescued by ST")
    for e in sorted(goals, key=lambda e: e["frame"]):
        lines.append("  {} {:<10} z={:<7} s={:<5} {:<5} {:<5} {:<6} {} {:<6} {:<5} {:<6} {}".format(
            e["half"], e["source"], str(e["z"]), round(e["strength"], 2), str(e["st_goal"]),
            str(e["st_kickoff_after_goal"]), str(e["to_kickoff_s"]),
            {0: "H", 1: "A"}.get(e["scorer"], "?"), str(e.get("kicker_s")),
            str(e.get("ball_seen_at_kickoff")), str(e.get("ball_after_kickoff_m")),
            str(e.get("kickoff_rescued_by_st"))))
    void = [e for e in eps if e.get("status") == "void"]
    lines.append("ball between the posts beyond the line with no kick-off after (void): {}  "
                 "z (p10/p50/p90): {}".format(len(void), _q([e["z"] for e in void])))
    lines.append("Smart Tagging GOALS with no goal found nearby: {}".format(
        sum(e.get("status") == "st_goal_without_goal" for e in eps)))
    lines.append("kick-offs skipped for missing attack direction: {}".format(
        sum(e.get("status") == "no_attack_dir" for e in eps)))
    return lines


def write_episodes(outputs, folder):
    """Episode lists WITH frame numbers -- local debugging only, never commit, never share."""
    import pandas as pd
    os.makedirs(folder, exist_ok=True)
    paths = []
    for rule_id, output in outputs.items():
        path = os.path.join(folder, "{}_episodes.csv".format(rule_id))
        pd.DataFrame(output.episodes).to_csv(path, index=False)
        paths.append(path)
    return paths
