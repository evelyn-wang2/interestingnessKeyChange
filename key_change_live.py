"""
Requirements:
    pip install essentia numpy mido python-rtmidi

Usage:
    python key_change_live.py
    python key_change_live.py --window 6 --confirm 2      # more responsive
    python key_change_live.py --changes-only              # print only on changes
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

try:
    import essentia.standard as ess
except ImportError:
    print("Missing 'essentia'. Install with:\n\n    pip install essentia numpy\n")
    sys.exit(1)

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# Scale definitions
# name -> (semitones above tonic, offset from tonic up to the parent major key)
DIATONIC: Dict[str, Tuple[Tuple[int, ...], int]] = {
    "major":      ((0, 2, 4, 5, 7, 9, 11), 0),
    "lydian":     ((0, 2, 4, 6, 7, 9, 11), 7),   # C lydian  = G major's notes
    "mixolydian": ((0, 2, 4, 5, 7, 9, 10), 5),   # C mixo    = F major's notes
    "dorian":     ((0, 2, 3, 5, 7, 9, 10), 10),  # D dorian  = C major's notes
    "minor":      ((0, 2, 3, 5, 7, 8, 10), 3),   # A minor   = C major's notes
    "phrygian":   ((0, 1, 3, 5, 7, 8, 10), 8),   # E phrygian= C major's notes
    "locrian":    ((0, 1, 3, 5, 6, 8, 10), 1),   # B locrian = C major's notes
}
# Essentia says major or minor; these are the modes each can be refined into.
FAMILIES: Dict[str, Tuple[str, ...]] = {
    "major": ("major", "lydian", "mixolydian"),
    "minor": ("minor", "dorian", "phrygian", "locrian"),
}
# name -> (pitch-class template, period): transpositions repeat every `period` semitones
SYMMETRIC: Dict[str, Tuple[Tuple[int, ...], int]] = {
    "diminished": ((0, 2, 3, 5, 6, 8, 9, 11), 3),
    "augmented":  ((0, 3, 4, 7, 8, 11), 4),
}
MOOD = {
    "major": "major (bright)",
    "lydian": "lydian (dreamy)",
    "mixolydian": "mixolydian (bluesy)",
    "dorian": "dorian (soulful)",
    "minor": "minor (dark)",
    "phrygian": "phrygian (brooding)",
    "locrian": "locrian (unstable)",
    "diminished": "diminished (tense)",
    "augmented": "augmented (unsettled)",
}
_MAJOR_SET = DIATONIC["major"][0]


class KeyState(NamedTuple):
    tonic: int                        # 0-11
    mode: str                         # key of MOOD
    ident: tuple                      # identity used to decide "is this a different key?"
    base: Optional[Tuple[int, str]]   # Essentia's (tonic, major/minor); None for symmetric scales


# Circle of fifths
def signature_position(state: KeyState) -> Optional[int]:
    """Slot 0-11 on the circle of fifths of the key signature this state
    uses (modes use their parent major scale). None for symmetric scales."""
    if state.mode in SYMMETRIC:
        return None
    parent = (state.tonic + DIATONIC[state.mode][1]) % 12
    return (parent * 7) % 12


def circle_steps(old: KeyState, new: KeyState) -> Optional[int]:
    """Shortest distance between two key signatures on the circle, 0-6.
    None if either has no key signature (diminished/augmented)."""
    a, b = signature_position(old), signature_position(new)
    if a is None or b is None:
        return None
    d = abs(a - b) % 12
    return min(d, 12 - d)


# Tracker
class KeyChangeTracker:
    def __init__(
        self,
        window_seconds: float = 4.0,
        hysteresis_margin: float = 0.05,
        confirm_hops: int = 3,
        min_notes: int = 3,
        profile: str = "bgate",
        recency_half_life: Optional[float] = 4.0,
        use_polyphony: bool = False,   # MIDI has no harmonics; False tested better
        mode_margin: float = 0.06,     # share of played notes a mode must explain beyond major/minor
    ):
        self.window_seconds = window_seconds
        self.hysteresis_margin = hysteresis_margin
        self.confirm_hops = max(1, confirm_hops)
        self.min_notes = min_notes
        self.recency_half_life = recency_half_life
        self.mode_margin = mode_margin
        self._algo = ess.Key(profileType=profile, usePolyphony=use_polyphony)

        self._done: List[Tuple[int, float, float]] = []   # (pitch, onset, duration)
        self._held: Dict[int, float] = {}                  # pitch -> onset
        self.current: Optional[KeyState] = None
        self._cand: Optional[KeyState] = None
        self._cand_hits = 0

    # input
    def note_on(self, pitch: int, t: float) -> None:
        self._held[pitch] = t

    def note_off(self, pitch: int, t: float) -> None:
        onset = self._held.pop(pitch, None)
        if onset is not None:
            self._done.append((pitch, onset, max(t - onset, 0.05)))

    # analysis
    def _histogram(self, now: float) -> Tuple[np.ndarray, int]:
        cutoff = now - self.window_seconds
        self._done = [n for n in self._done if n[1] + n[2] >= cutoff]
        notes = self._done + [(p, on, max(now - on, 0.05)) for p, on in self._held.items()]
        hist = np.zeros(12, dtype=np.float32)
        for pitch, onset, dur in notes:
            w = dur
            if self.recency_half_life:
                w *= 0.5 ** (max(now - onset, 0.0) / self.recency_half_life)
            hist[pitch % 12] += w
        return hist, len(notes)

    @staticmethod
    def _coverage(w: np.ndarray, tonic: int, intervals: Tuple[int, ...]) -> float:
        return float(sum(w[(tonic + i) % 12] for i in intervals))

    def _symmetric(self, w: np.ndarray) -> Optional[KeyState]:
        """Diminished/augmented: nearly all weight inside one collection, at
        least 5 of its notes actually used, and no major-scale set explains it."""
        if max(self._coverage(w, t, _MAJOR_SET) for t in range(12)) >= 0.90:
            return None
        best, best_cov = None, 0.0
        for name, (template, period) in SYMMETRIC.items():
            for t in range(period):
                pcs = [(t + i) % 12 for i in template]
                cov = float(w[pcs].sum())
                used = sum(1 for p in pcs if w[p] >= 0.03)
                if cov >= 0.95 and used >= 5 and cov > best_cov:
                    tonic = max(pcs, key=lambda p: w[p])
                    best, best_cov = KeyState(tonic, name, ("sym", name, t), None), cov
        return best

    def _classify(self, hist: np.ndarray) -> Optional[Tuple[KeyState, float]]:
        """Returns (state, essentia_gap). gap is inf for symmetric scales."""
        w = hist / hist.sum()
        sym = self._symmetric(w)
        if sym is not None:
            return sym, float("inf")
        try:
            # Essentia's pitch-class vectors start at A, not C -> rotate by 3.
            tonic_name, scale, _strength, gap = self._algo(np.roll(hist, 3))
        except RuntimeError:
            return None
        tonic = PITCH_CLASS_NAMES.index(_FLAT_TO_SHARP.get(tonic_name, tonic_name))

        family = FAMILIES[scale]
        cov = {m: self._coverage(w, tonic, DIATONIC[m][0]) for m in family}
        challenger = max((m for m in family if m != scale), key=cov.get)
        mode = challenger if cov[challenger] - cov[scale] >= self.mode_margin else scale
        return KeyState(tonic, mode, ("dia", tonic, mode), (tonic, scale)), float(gap)

    # -- interestingness -----------------------------------------------------
    def interestingness(self, steps: Optional[int]) -> str:
        if steps < 3:
            interestingnessLevel = "Standard"
        elif steps < 5:
            interestingnessLevel = "Medium"
        else:
            interestingnessLevel = "Interesting"
        return interestingnessLevel

    # output
    def estimate(self, now: float):
        """None until enough notes, else (state, changed, steps, is_start)."""
        hist, n = self._histogram(now)
        if n < self.min_notes or not hist.any():
            return None
        result = self._classify(hist)
        if result is None:
            return None
        cand, gap = result

        changed, steps, start = False, None, False
        if self.current is None:
            self.current, changed, start = cand, True, True
        elif cand.ident == self.current.ident:
            if cand.base is not None:      # keep symmetric scales' original tonic
                self.current = cand
            self._cand, self._cand_hits = None, 0
        else:
            # A different Essentia key must also beat the runner-up by a margin;
            # mode-only changes and symmetric scales only need to persist.
            needs_margin = (cand.base is not None and self.current.base is not None
                            and cand.base != self.current.base)
            if needs_margin and gap <= self.hysteresis_margin:
                self._cand, self._cand_hits = None, 0
            else:
                self._cand_hits = self._cand_hits + 1 if (self._cand and cand.ident == self._cand.ident) else 1
                self._cand = cand
                if self._cand_hits >= self.confirm_hops:
                    steps = circle_steps(self.current, cand)
                    self.current, changed = cand, True
                    self._cand, self._cand_hits = None, 0
        return self.current, changed, steps, start


# MIDI + terminal
def select_port(mido) -> str:
    try:
        names = mido.get_input_names()
    except Exception as exc:
        print(f"Could not query the system's MIDI ports ({exc}).")
        sys.exit(1)
    if not names:
        print("No MIDI input port detected. Connect your controller and try again.")
        sys.exit(1)
    if len(names) == 1:
        print(f"Using the only MIDI port detected: {names[0]}")
        return names[0]
    print("MIDI ports detected:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    while True:
        raw = input("Select the port number to use: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(names):
            return names[int(raw)]
        print("Invalid input, try again.")


HEADER = f"{'t(s)':>6} | {'key':<14} | {'mood':<21} | {'key change':<10} | {'circle steps':<12} | interestingness"
RULE = "-" * len(HEADER)


def format_row(now: float, state: KeyState, changed: bool, steps: Optional[int], start: bool,
               level: str = "") -> str:
    name = f"{PITCH_CLASS_NAMES[state.tonic]} {state.mode}"
    flag, step_txt = "", ""
    if start:
        flag = "(start)"
    elif changed:
        flag = "KEY CHANGE"
        step_txt = "n/a" if steps is None else str(steps)
    return (f"{now:6.1f} | {name:<14} | {MOOD[state.mode]:<21} | {flag:<10} | "
            f"{step_txt:<12} | {level}")


def run(port, tracker: KeyChangeTracker, hop: float, changes_only: bool,
        duration: Optional[float] = None) -> None:
    print(HEADER)
    print(RULE)
    start = time.time()
    last = 0.0
    while True:
        now = time.time() - start
        if duration is not None and now > duration:
            return
        for msg in port.iter_pending():
            if msg.type == "note_on" and msg.velocity > 0:
                tracker.note_on(msg.note, now)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                tracker.note_off(msg.note, now)
        if now - last >= hop:
            result = tracker.estimate(now)
            if result is not None:
                state, changed, steps, is_start = result
                if changed or not changes_only:
                    level = tracker.interestingness(steps) if (changed and not is_start) else ""
                    print(format_row(now, state, changed, steps, is_start, level), flush=True)
            last = now
        time.sleep(0.01)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live MIDI key/mode/key-change detector")
    ap.add_argument("--window", type=float, default=4.0, help="seconds of notes to consider (default 4)")
    ap.add_argument("--hop", type=float, default=0.5, help="seconds between rows (default 0.5)")
    ap.add_argument("--margin", type=float, default=0.05, help="Essentia hysteresis margin (default 0.05)")
    ap.add_argument("--confirm", type=int, default=3, help="estimates a new key must win in a row (default 3)")
    ap.add_argument("--mode-margin", type=float, default=0.06,
                    help="how much more of the played notes a mode must explain than plain "
                         "major/minor (default 0.06; higher = fewer modes reported)")
    ap.add_argument("--profile", default="bgate",
                    choices=["temperley", "krumhansl", "shaath", "edma", "bgate"])
    ap.add_argument("--changes-only", action="store_true", help="print a row only when the key changes")
    args = ap.parse_args()

    try:
        import mido
    except ImportError:
        print("Missing 'mido'. Install with:\n\n    pip install mido python-rtmidi\n")
        sys.exit(1)

    port_name = select_port(mido)
    tracker = KeyChangeTracker(
        window_seconds=args.window,
        hysteresis_margin=args.margin,
        confirm_hops=args.confirm,
        profile=args.profile,
        mode_margin=args.mode_margin,
    )
    print(f"Essentia key detection ({args.profile}), window={args.window}s, "
          f"confirm={args.confirm} hops. Ctrl+C to quit.\n")
    with mido.open_input(port_name) as port:
        try:
            run(port, tracker, args.hop, args.changes_only)
        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()