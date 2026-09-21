"""Transcription stage: mix the tracks, slice on silence, send each slice."""
from __future__ import annotations

from geshtu import audio
from geshtu import engines
from geshtu.models import Segment

TARGET = 120.0          # seconds aimed at per slice


class Transcribe:
    name = "transcribe"
    provides = ("segments",)

    def run(self, session, cfg, report) -> None:
        if not session.mixed or not session.mixed.exists():
            session.mixed = session.root / "mixed.wav"
            # ⚠️ EVERY track, taken from the session itself rather than from a
            # hardcoded list of kinds. A list went stale once and silently
            # dropped a whole track at mix time while its per-segment level
            # was still being reported as healthy.
            if not audio.mix([t.segments for t in session.tracks],
                             session.mixed, cfg.rate):
                report("no audio to transcribe")
                return

        if audio.is_silent(session.mixed):
            report("the mix is silent -- refusing to transcribe")
            return

        total = audio.duration(session.mixed)
        cuts = audio.silence_cuts(session.mixed, total and "-30dB" or "-30dB")
        bounds = _bounds(cuts, total)
        report("%d slices over %.0f s" % (len(bounds), total))

        engine = cfg.engine("asr")
        work = session.root / "slices"
        work.mkdir(exist_ok=True)
        session.segments = []
        for i, (start, end) in enumerate(bounds):
            piece = audio.slice_out(session.mixed, work / ("%03d.wav" % i), start, end,
                                    cfg.rate)
            text = engines.transcribe(engine, piece)
            if text:
                session.segments.append(Segment(start=start, end=end, text=text))
            report("  slice %d/%d  %.0f-%.0f s  %d words"
                   % (i + 1, len(bounds), start, end, len(text.split())))


def _bounds(cuts: list[float], total: float) -> list[tuple[float, float]]:
    """Group silence points into slices of roughly TARGET seconds.

    ⚠️ Cutting at a fixed interval slices through a word and the engine
    returns two confident halves of nothing. Always cut where nobody speaks.
    """
    if total <= TARGET or not cuts:
        return [(0.0, total)]
    bounds, start = [], 0.0
    for c in cuts:
        if c - start >= TARGET:
            bounds.append((start, c))
            start = c
    if total - start > 1.0:
        bounds.append((start, total))
    return bounds or [(0.0, total)]


PLUGIN = Transcribe
