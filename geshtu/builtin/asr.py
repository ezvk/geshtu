"""Transcription stage: mix the tracks, slice on silence, send each slice."""
from __future__ import annotations

from geshtu import audio
from geshtu import engines
from geshtu.models import Segment

TARGET = 120.0          # seconds aimed at per slice


class Transcribe:
    name = "transcribe"
    requires = ()
    provides = ("segments",)

    def run(self, session, cfg, report) -> None:
        if not audio.ensure_mixed(session, cfg, report):
            return
        if audio.is_silent(session.mixed):
            report("the mix is silent -- refusing to transcribe")
            return

        total = audio.duration(session.mixed)
        cuts = audio.silence_cuts(session.mixed)
        bounds = _bounds(cuts, total, session.turns)
        report("%d slices over %.0f s" % (len(bounds), total))

        engine = cfg.engine("asr")
        work = session.root / "slices"
        work.mkdir(exist_ok=True)
        session.segments = []
        for i, (start, end) in enumerate(bounds):
            piece = audio.slice_out(session.mixed, work / ("%03d.wav" % i),
                                    start, end, cfg.rate)
            text = engines.transcribe(engine, piece)
            if not text:
                continue
            seg = Segment(start=start, end=end, text=text)
            seg.speaker = _qui(session.turns, start, end)
            session.segments.append(seg)
            report("  slice %d/%d  %.0f-%.0f s  %d words%s"
                   % (i + 1, len(bounds), start, end, len(text.split()),
                      "  " + seg.speaker if seg.speaker else ""))


def _qui(turns, start, end):
    """Le locuteur qui occupe le plus cette tranche.

    ⚠️ AU PLUS GRAND RECOUVREMENT, pas au premier qui commence : une tranche
    coupee sur un silence commence regulierement dans la traine du precedent.
    """
    best, meilleur = None, 0.0
    for t in turns:
        part = min(end, t.end) - max(start, t.start)
        if part > meilleur:
            best, meilleur = t.speaker, part
    return best


def _bounds(cuts: list[float], total: float,
            turns=()) -> list[tuple[float, float]]:
    """Group silence points into slices of roughly TARGET seconds.

    ⚠️ CUT ON SILENCE, NEVER AT A FIXED INTERVAL: a fixed cut slices through
    a word and the engine returns two confident halves of nothing.

    ⚠️ AND CUT AT EVERY CHANGE OF VOICE, whatever the length. ezvk, on a
    conference recording: "là y a des questions du public". A question lasts
    a few seconds inside a stretch of lecture; with slices cut only on
    silence, it lands in the middle of a two-minute block and the speaker
    who occupies most of that block takes the credit. Measured before this
    change: three speakers found, one single label in the whole note.
    """
    if total <= 0:
        return []
    frontieres = {0.0, total}
    for t in turns:
        # ⚠️ Les bornes de tour, meme tres rapprochees : c est exactement ce
        # qu on veut conserver ici.
        if 0.0 < t.start < total:
            frontieres.add(round(t.start, 3))
        if 0.0 < t.end < total:
            frontieres.add(round(t.end, 3))

    if not turns:
        if total <= TARGET or not cuts:
            return [(0.0, total)]
        bornes, debut = [], 0.0
        for c in cuts:
            if c - debut >= TARGET:
                bornes.append((debut, c))
                debut = c
        if total - debut > 1.0:
            bornes.append((debut, total))
        return bornes or [(0.0, total)]

    # avec des tours : on decoupe d abord par locuteur, puis on aere les
    # longues plages sur les silences pour ne pas depasser le plafond du
    # moteur de transcription
    points = sorted(frontieres)
    bornes = []
    for a, b in zip(points, points[1:]):
        if b - a < 0.2:
            continue
        debut = a
        for c in cuts:
            if a < c < b and c - debut >= TARGET:
                bornes.append((debut, c))
                debut = c
        if b - debut > 0.2:
            bornes.append((debut, b))
    return bornes or [(0.0, total)]


PLUGIN = Transcribe
