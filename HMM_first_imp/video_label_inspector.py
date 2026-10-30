"""Interactive frame-by-frame inspector: a match video, the ground truth and a model's labels, in sync.

    python src/tikitalia/video_label_inspector.py
    python src/tikitalia/video_label_inspector.py --model evaluation/results/baseline_raw.csv
    python src/tikitalia/video_label_inspector.py --video path/to/video.mp4 --labels path/to/labels.csv

Row i of a CSV (columns `frame,label`) is frame i of the video. Labels are H (home in possession),
A (away in possession), D (dead ball) and X (nothing / no ground truth). The defaults are the tactical
video, the public ground truth and tikitalia.csv next to this script (shown only if it exists). The
public ground truth only covers frames 11185..18684, so the inspector opens on the first frame it
labels unless --start says otherwise; elsewhere the model has nothing to be compared with.

On the tactical video a side panel lists the Smart Tagging codes active on the current frame. Its
intervals overlap (up to 12 at once), so a frame has a list of codes rather than one code.

Read-only by design: it displays labels and cannot edit them. RULES.md allows watching the video to
debug a pipeline, but not turning what you see into labels or corrections on the provided game.

Controls:
    Right / Left    +1 / -1 frame            Space       play / pause
    D / A           +10 / -10 frames         [ / ]       slower / faster playback
    Up / Down       +100 / -100 frames       N / P       next / previous label change (either file)
    PgUp / PgDn     +1000 / -1000 frames     0-9 Enter   type a frame number and jump to it
    Home / End      first / last frame       Q / Esc     quit

Requires OpenCV: pip install opencv-python
"""
import argparse
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]

from games import GAME, half_of_video_frame, video_path  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

DEFAULT_LABELS = os.path.join(ROOT, "evaluation", "gt", f"{GAME}_possession_gt_public.csv")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tikitalia.csv")
WINDOW = "Video label inspector"
TRACKBAR = "frame"

LABELS = "HADX"
H, A, D, X = range(4)  # integer code of each label: its position in LABELS
LABEL_NAMES = ("Barcelona", "Athletic Club", "Dead ball", "Nothing")

# All colours are BGR, as OpenCV expects. The badge uses the team's two colours: (fill, side stripes).
BADGE_COLOURS = (
    ((68, 0, 165), (152, 77, 0)),  # H: blaugrana, garnet with blue stripes
    ((19, 6, 227), (255, 255, 255)),  # A: red with white stripes
    ((128, 128, 128), (128, 128, 128)),  # D: grey
    ((0, 0, 0), (255, 255, 255)),  # X: black with a white outline
)
# On a timeline a pixel holds one colour only, and garnet next to red would be unreadable: H is blue.
TIMELINE_COLOURS = np.array(
    [(152, 77, 0), (19, 6, 227), (128, 128, 128), (0, 0, 0)], dtype=np.uint8
)

PANEL_BG = (32, 32, 32)
TEXT = (235, 235, 235)
MUTED = (165, 165, 165)
FONT = cv2.FONT_HERSHEY_SIMPLEX

PANEL_HEIGHT = 136
BADGE_SIZE = 112
TAG_PANEL_WIDTH = 240
# The Team tag of a Smart Tagging instance, as a side. Spelled as in the XML; N/A is not a team.
TAG_TEAMS = {"Barcelona": H, "Atletic Club": A}
LOCAL_SPAN_S = 10  # the zoomed timeline shows this many seconds on each side of the current frame
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)

# The command bar under the panel, as (keys, action). It wraps to the width of the window.
COMMANDS = (
    ("Left / Right", "-1 / +1 frame"),
    ("A / D", "-10 / +10"),
    ("Down / Up", "-100 / +100"),
    ("PgDn / PgUp", "-1000 / +1000"),
    ("Home / End", "inizio / fine"),
    ("Space", "play / pausa"),
    ("[ / ]", "piu lento / piu veloce"),
    ("P / N", "cambio etichetta prec. / succ."),
    ("0-9 + Enter", "vai al frame"),
    ("Q / Esc", "esci"),
)

# Arrow-like keys have different codes on each platform; waitKeyEx returns them unmasked.
KEY_LEFT = {2424832, 65361, 63234}
KEY_RIGHT = {2555904, 65363, 63235}
KEY_UP = {2490368, 65362, 63232}
KEY_DOWN = {2621440, 65364, 63233}
KEY_PAGE_UP = {2162688, 65365, 63276}
KEY_PAGE_DOWN = {2228224, 65366, 63277}
KEY_HOME = {2359296, 65360, 63273}
KEY_END = {2293760, 65367, 63275}
KEY_ENTER = {10, 13}
KEY_BACKSPACE = {8, 127}
KEY_ESC = 27


def load_labels(path):
    """Read a `frame,label` CSV into an array of label codes, indexed by frame.

    Frames the file does not mention become X. Anything ambiguous is an error rather than a guess,
    because a silently shifted label is exactly what this tool exists to catch.
    """
    table = pd.read_csv(path, dtype={"label": str}, keep_default_na=False)
    missing = {"frame", "label"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}, found {list(table.columns)}")
    if table.empty:
        raise ValueError(f"{path}: no rows")
    if table["frame"].duplicated().any():
        raise ValueError(f"{path}: duplicate frame numbers")
    if table["frame"].min() < 0:
        raise ValueError(f"{path}: negative frame numbers")

    labels = table["label"].str.strip().str.upper()
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"{path}: unknown label(s) {unknown}, expected one of {list(LABELS)}")

    codes = np.full(int(table["frame"].max()) + 1, X, dtype=np.uint8)
    codes[table["frame"].to_numpy()] = labels.map(LABELS.index).to_numpy(dtype=np.uint8)
    return codes


def fit_labels(codes, n_frames, name="labels"):
    """Make the labels exactly as long as the video, warning when they disagree."""
    if len(codes) != n_frames:
        print(f"WARNING: {name} cover {len(codes)} frames, the video has {n_frames}. "
              "Is this the right video? (submissions follow the tactical one)", file=sys.stderr)
    fitted = np.full(n_frames, X, dtype=np.uint8)
    n = min(len(codes), n_frames)
    fitted[:n] = codes[:n]
    return fitted


def build_command_bar(width):
    """Draw the keyboard commands once: they never change, so each frame only has to paste them."""
    font_scale, row_height, margin = 0.45, 28, 8
    # Lay the commands out first, wrapping to a new row when one would not fit.
    placed, x, row = [], margin, 0
    for keys, action in COMMANDS:
        keys_w = cv2.getTextSize(keys, FONT, font_scale, 1)[0][0] + 10
        action_w = cv2.getTextSize(action, FONT, font_scale, 1)[0][0]
        if x > margin and x + keys_w + 6 + action_w > width - margin:
            x, row = margin, row + 1
        placed.append((keys, action, x, keys_w, row))
        x += keys_w + 6 + action_w + 16

    bar = np.full(((row + 1) * row_height + 6, width, 3), (55, 55, 55), dtype=np.uint8)
    cv2.line(bar, (0, 0), (width - 1, 0), MUTED, 1)
    for keys, action, x, keys_w, row in placed:
        top = 6 + row * row_height
        # Each key looks like a key cap: light box, dark text.
        cv2.rectangle(bar, (x, top), (x + keys_w, top + 20), (225, 225, 225), -1)
        cv2.putText(bar, keys, (x + 5, top + 15), FONT, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(bar, action, (x + keys_w + 6, top + 15), FONT, font_scale, TEXT, 1, cv2.LINE_AA)
    return bar


class FrameReader:
    """Random access to video frames, with a cache that makes stepping backwards instantaneous.

    A compressed video can only be decoded forwards from a keyframe, so a seek costs far more than
    the next frame. Hence: consecutive frames never seek, short forward jumps decode through, and a
    short step back decodes a whole chunk ending at the requested frame, so the next steps back are
    served from the cache. Frames are cached at display size to keep the memory bounded.
    """

    BACKFILL = 32  # frames decoded in one go when stepping backwards
    MAX_DECODE_THROUGH = 64  # beyond this forward distance a seek is cheaper than decoding through

    def __init__(self, path, width, cache_frames):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        src_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self.n_frames <= 0 or src_w <= 0 or src_h <= 0:
            raise OSError(f"Cannot read frame count or size of: {path}")
        self.size = (width, round(src_h * width / src_w))
        self.cache = OrderedDict()
        self.cache_frames = max(cache_frames, self.BACKFILL + 1)
        self.next_pos = 0  # the frame the decoder would return on the next read

    def _seek(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.next_pos = frame

    def _decode_next(self):
        """Decode the next frame into the cache and return it, or None at a decoding failure."""
        ok, image = self.cap.read()
        if not ok:
            return None
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self.cache[self.next_pos] = image
        self.cache.move_to_end(self.next_pos)
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        self.next_pos += 1
        return image

    def get(self, frame):
        """Return one frame at display size, or None if it cannot be decoded."""
        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]

        distance = frame - self.next_pos
        if distance < 0 and -distance <= 2 * self.BACKFILL:
            self._seek(max(0, frame - self.BACKFILL + 1))
        elif distance < 0 or distance > self.MAX_DECODE_THROUGH:
            self._seek(frame)

        # Skip without decoding to pixels, unless the frames are wanted for the cache (backfill).
        backfilling = distance < 0
        while self.next_pos < frame:
            if backfilling:
                if self._decode_next() is None:
                    return None
            else:
                if not self.cap.grab():
                    return None
                self.next_pos += 1
        return self._decode_next()

    def release(self):
        self.cap.release()


class Inspector:
    """The state of the session (current frame, play/pause) and how it is drawn."""

    def __init__(self, reader, tracks, start, tags=()):
        self.reader = reader
        # Sorted by start, so that a code keeps its relative place in the list while it is active.
        self.tags = sorted(tags, key=lambda tag: tag["start_frame"])
        self.tag_starts = np.array([tag["start_frame"] for tag in self.tags], dtype=int)
        self.tag_ends = np.array([tag["end_frame"] for tag in self.tags], dtype=int)
        self.tracks = tracks  # [(name, label codes)]: the reference first, then the model if any
        self.last = reader.n_frames - 1
        self.frame = int(np.clip(start, 0, self.last))
        self.playing = False
        self.speed_index = SPEEDS.index(1.0)
        self.typed = ""  # digits typed so far for a jump to a frame number
        self.pending_seek = None  # position requested by the trackbar, applied by the main loop
        self.status = ""

        # Where each run of equal labels starts, per track. N / P stop at a change in any track: a
        # disagreement can only begin at one, so none can be stepped over.
        self.run_starts = [np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1)) for _, codes in tracks]
        self.change_starts = np.unique(np.concatenate(self.run_starts))
        everywhere = np.arange(reader.size[0]) * reader.n_frames // reader.size[0]
        self.full_strips = [self._timeline_strip(codes, everywhere) for _, codes in tracks]
        self.command_bar = build_command_bar(reader.size[0])

    # -- navigation --------------------------------------------------------------------------------

    def go_to(self, frame):
        self.frame = int(np.clip(frame, 0, self.last))

    def next_change(self):
        """Jump to the first frame of the next run of labels."""
        i = np.searchsorted(self.change_starts, self.frame, side="right")
        if i < len(self.change_starts):
            self.go_to(self.change_starts[i])
        else:
            self.status = "No later label change"

    def previous_change(self):
        """Jump to the start of the current run, or of the previous one if already at the start."""
        i = np.searchsorted(self.change_starts, self.frame, side="left") - 1
        if i >= 0:
            self.go_to(self.change_starts[i])
        else:
            self.status = "No earlier label change"

    def handle_key(self, key):
        """Apply one key press. Returns False when the user asked to quit."""
        char = chr(key).lower() if 0 <= key < 256 else ""

        if "0" <= char <= "9":
            self.typed += char
            return True
        if key in KEY_BACKSPACE:
            self.typed = self.typed[:-1]
            return True
        if key in KEY_ENTER:
            if self.typed:
                self.go_to(int(self.typed))
                self.typed = ""
            return True
        if key == KEY_ESC and self.typed:  # Esc first cancels a number being typed, then quits
            self.typed = ""
            return True
        if key == KEY_ESC or char == "q":
            return False

        if char == " ":
            self.playing = not self.playing
            return True
        if char == "[":
            self.speed_index = max(self.speed_index - 1, 0)
            return True
        if char == "]":
            self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
            return True

        # Any other navigation is a request to look at a precise frame, so it pauses the playback.
        if key in KEY_RIGHT:
            target = self.frame + 1
        elif key in KEY_LEFT:
            target = self.frame - 1
        elif char == "d":
            target = self.frame + 10
        elif char == "a":
            target = self.frame - 10
        elif key in KEY_UP:
            target = self.frame + 100
        elif key in KEY_DOWN:
            target = self.frame - 100
        elif key in KEY_PAGE_UP:
            target = self.frame + 1000
        elif key in KEY_PAGE_DOWN:
            target = self.frame - 1000
        elif key in KEY_HOME:
            target = 0
        elif key in KEY_END:
            target = self.last
        elif char == "n":
            self.playing = False
            self.next_change()
            return True
        elif char == "p":
            self.playing = False
            self.previous_change()
            return True
        else:
            return True  # an unbound key, or the -1 of a waitKey timeout
        self.playing = False
        self.go_to(target)
        return True

    # -- drawing -----------------------------------------------------------------------------------

    def _timeline_strip(self, codes, frames):
        """One row of timeline colours for the given frames; frames outside the video are dark."""
        inside = (frames >= 0) & (frames < len(codes))
        strip = np.full((1, len(frames), 3), PANEL_BG, dtype=np.uint8)
        strip[0, inside] = TIMELINE_COLOURS[codes[frames[inside]]]
        return strip

    def _draw_timeline(self, canvas, top, row_height, strips, cursor_x, caption):
        """One row per track, one above the other so that the same x is the same frame."""
        width = canvas.shape[1]
        for row, strip in enumerate(strips):
            row_top = top + row * (row_height + 2)
            canvas[row_top:row_top + row_height] = strip
            cv2.rectangle(canvas, (0, row_top), (width - 1, row_top + row_height - 1), MUTED, 1)
        bottom = top + len(strips) * (row_height + 2)
        cv2.line(canvas, (cursor_x, top - 3), (cursor_x, bottom), (0, 255, 255), 2)
        cv2.putText(canvas, caption, (8, top - 5), FONT, 0.4, MUTED, 1, cv2.LINE_AA)

    def _draw_badge(self, canvas, x, y, name, code):
        """One label in the colours of its team, with whose label it is above and its meaning below."""
        fill, stripe = BADGE_COLOURS[code]
        w, h, side, band = BADGE_SIZE, BADGE_SIZE, 14, 22
        cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + side, y + h), stripe, -1)
        cv2.rectangle(canvas, (x + w - side, y), (x + w, y + h), stripe, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + band), (0, 0, 0), -1)
        cv2.rectangle(canvas, (x, y + h - band), (x + w, y + h), (0, 0, 0), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 2)

        (text_w, text_h), _ = cv2.getTextSize(LABELS[code], FONT, 2.0, 4)
        origin = (x + (w - text_w) // 2, y + (h + text_h) // 2)
        cv2.putText(canvas, LABELS[code], origin, FONT, 2.0, (255, 255, 255), 4, cv2.LINE_AA)

        for text, baseline, scale in ((name, y + 16, 0.5), (LABEL_NAMES[code], y + h - 7, 0.42)):
            (text_w, _), _ = cv2.getTextSize(text, FONT, scale, 1)
            origin = (x + (w - text_w) // 2, baseline)
            cv2.putText(canvas, text, origin, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_badges(self, canvas):
        """All the labels of this frame, top-right over the video, the reference on the left.

        Not top-left: that corner holds the match clock and the score burnt into the video.
        """
        codes = [int(track[self.frame]) for _, track in self.tracks]
        x = canvas.shape[1] - 16 - len(codes) * (BADGE_SIZE + 8) + 8
        left = x
        for (name, _), code in zip(self.tracks, codes):
            self._draw_badge(canvas, x, 16, name, code)
            x += BADGE_SIZE + 8

        # The captions sit on a solid box: an outline alone gets lost against the bright stands.
        captions =[(f"frame {self.frame}", (255, 255, 255))]
        if len(codes) > 1:
            if codes[0] == X:
                captions.append(("no ground truth here", (200, 200, 200)))
            elif codes[0] == codes[1]:
                captions.append(("AGREE", (90, 255, 90)))
            else:
                captions.append(("DISAGREE", (80, 80, 255)))
        for line, (caption, colour) in enumerate(captions):
            (text_w, _), _ = cv2.getTextSize(caption, FONT, 0.8, 2)
            origin = (left - 18 - text_w, 46 + line * 36)
            cv2.rectangle(canvas, (origin[0] - 8, origin[1] - 24), (left - 10, origin[1] + 8), (0, 0, 0), -1)
            cv2.putText(canvas, caption, origin, FONT, 0.8, colour, 2, cv2.LINE_AA)

    def _draw_tag_panel(self, height):
        """The Smart Tagging codes active on this frame (end_frame is exclusive), one per row."""
        panel = np.full((height, TAG_PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, height - 1), MUTED, 1)
        active = np.flatnonzero((self.tag_starts <= self.frame) & (self.frame < self.tag_ends))
        cv2.putText(panel, f"SMART TAGGING ({len(active)})", (12, 26), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
        cv2.putText(panel, "H / A = Team tag, not possession", (12, 46), FONT, 0.4, MUTED, 1, cv2.LINE_AA)
        if len(active) == 0:
            cv2.putText(panel, "no code on this frame", (12, 76), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

        for row, index in enumerate(active):
            tag = self.tags[index]
            team = TAG_TEAMS.get(tag["tags"].get("Team"))
            top = 60 + row * 26
            chip = tuple(int(c) for c in TIMELINE_COLOURS[team]) if team is not None else (90, 90, 90)
            cv2.rectangle(panel, (12, top), (32, top + 20), chip, -1)
            letter = LABELS[team] if team is not None else "-"
            cv2.putText(panel, letter, (17, top + 15), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, tag["code"], (40, top + 15), FONT, 0.42, TEXT, 1, cv2.LINE_AA)
        return panel

    def render(self):
        """Compose the current frame and the information panels into one image."""
        width, height = self.reader.size
        image = self.reader.get(self.frame)
        canvas = np.full((height + PANEL_HEIGHT, width, 3), PANEL_BG, dtype=np.uint8)
        if image is None:
            canvas[:height] = 0
            cv2.putText(canvas, f"Frame {self.frame} could not be decoded", (40, height // 2),
                        FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            canvas[:height] = image

        self._draw_badges(canvas)

        # Line of text: where we are in the video and in the current run of each track.
        total_s = self.frame / self.reader.fps
        half = half_of_video_frame(GAME, self.frame)
        phase = f"half {half}" if half else "outside play"
        state = f"PLAY x{SPEEDS[self.speed_index]:g}" if self.playing else "PAUSED"
        info = f"{self.frame} / {self.last}   {int(total_s // 60):02d}:{total_s % 60:05.2f}   {phase}"
        for (name, codes), starts in zip(self.tracks, self.run_starts):
            run = np.searchsorted(starts, self.frame, side="right") - 1
            run_end = starts[run + 1] if run + 1 < len(starts) else len(codes)
            info += (f"   |   {name} {LABELS[codes[self.frame]]}: "
                     f"{self.frame - starts[run] + 1} of {run_end - starts[run]}")
        info += f"   |   {state}"
        cv2.putText(canvas, info, (8, height + 20), FONT, 0.5, TEXT, 1, cv2.LINE_AA)
        if self.typed:
            prompt = f"go to frame: {self.typed}_  (Enter to jump, Esc to cancel)"
            cv2.putText(canvas, prompt, (16, height - 16), FONT, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, prompt, (16, height - 16), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        elif self.status:
            cv2.putText(canvas, self.status, (16, height - 16), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Two timelines: zoomed around the current frame (centred cursor), and the whole video.
        span = int(LOCAL_SPAN_S * self.reader.fps)
        around = self.frame - span + np.arange(width) * (2 * span + 1) // width
        order = " over ".join(name for name, _ in self.tracks)
        strips = [self._timeline_strip(codes, around) for _, codes in self.tracks]
        self._draw_timeline(canvas, height + 44, 13, strips, width // 2,
                            f"+/- {LOCAL_SPAN_S} s   ({order})")
        cursor = self.frame * width // self.reader.n_frames
        self._draw_timeline(canvas, height + 92, 7, self.full_strips, cursor, "whole video")

        colours = "timeline:  H blue   A red   D grey   X black"
        cv2.putText(canvas, colours, (8, height + 128), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
        canvas = np.vstack((canvas, self.command_bar))
        if self.tags:
            canvas = np.hstack((canvas, self._draw_tag_panel(canvas.shape[0])))
        return canvas

    # -- main loop ---------------------------------------------------------------------------------

    def _on_trackbar(self, position):
        # Moving the trackbar from code fires this callback too: only a real drag differs from frame.
        if position != self.frame:
            self.pending_seek = position

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(TRACKBAR, WINDOW, self.frame, self.last, self._on_trackbar)
        frame_duration = 1.0 / self.reader.fps
        shown = None
        next_tick = time.perf_counter()

        while True:
            if self.pending_seek is not None:
                self.go_to(self.pending_seek)
                self.pending_seek = None
                self.playing = False

            state = (self.frame, self.playing, self.speed_index, self.typed, self.status)
            if state != shown:  # redrawing an unchanged view would only waste a decode and a blit
                cv2.imshow(WINDOW, self.render())
                cv2.setTrackbarPos(TRACKBAR, WINDOW, self.frame)
                shown = state

            # While playing, wait for what is left of this frame's time slot, minus a margin: on
            # Windows waitKey can overshoot by a timer tick (~16 ms), which at 25 fps loses frames.
            if self.playing:
                delay_ms = max(1, int((next_tick - time.perf_counter()) * 1000) - 16)
            else:
                delay_ms = 30
            key = cv2.waitKeyEx(delay_ms)

            if key != -1:
                self.status = ""
                if not self.handle_key(key):
                    break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # closed with the X button
                break

            if not self.playing:
                next_tick = time.perf_counter()
            elif time.perf_counter() >= next_tick:
                if self.frame >= self.last:
                    self.playing = False
                else:
                    # Keep to real time: if drawing cannot keep up (fast speeds), skip frames on
                    # screen. To check every single frame, pause and step with the arrows.
                    duration = frame_duration / SPEEDS[self.speed_index]
                    steps = 1 + int((time.perf_counter() - next_tick) / duration)
                    self.go_to(self.frame + steps)
                    next_tick += steps * duration
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a match video synchronised with per-frame labels.")
    parser.add_argument("--video", help="video file (default: the tactical video of the game)")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="reference CSV with columns frame,label (default: the public ground truth)")
    parser.add_argument("--model",
                        help="model CSV with columns frame,label (default: tikitalia.csv next to this script)")
    parser.add_argument("--start", type=int, help="first frame to show (default: the first labelled frame)")
    parser.add_argument("--width", type=int, default=1280, help="display width in pixels (default: 1280)")
    parser.add_argument("--cache-frames", type=int, default=150,
                        help="decoded frames kept in memory (default: 150, about 400 MB at 1280 px)")
    args = parser.parse_args(argv)

    video = args.video or video_path()
    # A model asked for by name must exist; the default one is simply left out when it does not.
    model = args.model or (DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else None)
    for path in (video, args.labels, model):
        if path and not os.path.exists(path):
            parser.error(f"file not found: {path}")

    try:
        reader = FrameReader(video, args.width, args.cache_frames)
        tracks = [("GT", fit_labels(load_labels(args.labels), reader.n_frames, "GT labels"))]
        if model:
            tracks.append(("MODEL", fit_labels(load_labels(model), reader.n_frames, "model labels")))
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")

    # Smart Tagging is on the tactical timeline: on any other video it would show the wrong codes.
    tags = []
    if os.path.abspath(video) == os.path.abspath(video_path()):
        try:
            tags = load_tagging()
        except OSError as error:
            print(f"WARNING: no Smart Tagging panel. {error}", file=sys.stderr)

    start = args.start
    if start is None:
        labelled = np.flatnonzero(tracks[0][1] != X)
        start = int(labelled[0]) if len(labelled) else 0

    print(f"{os.path.basename(video)}: {reader.n_frames} frames at {reader.fps:g} fps")
    for name, codes in tracks:
        counts = "  ".join(f"{label}={int((codes == i).sum())}" for i, label in enumerate(LABELS))
        print(f"  {name}: {counts}")
    print(f"  Smart Tagging: {len(tags)} intervals")
    try:
        Inspector(reader, tracks, start, tags).run()
    finally:
        reader.release()


if __name__ == "__main__":
    main()
