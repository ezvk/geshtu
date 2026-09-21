"""Chaptering by topic, using embeddings — and why it is not optional.

⚠️ THE ACCELERATOR HAS A HARD PROMPT CEILING. Measured on an Intel NPU
(2026-09-19) by bisection: 8192 tokens compiles in 75 s, 9216 is refused in
15 s, and so is 16384. A lighter model does not help -- Mistral-7B, a gigabyte
smaller, fails at exactly the same 9216. It is a compiler constant, not a
memory limit, so it cannot be raised by configuration or firmware.

One hour of speech is 10,000 to 15,000 tokens. The whole transcript therefore
NEVER fits. Chaptering is what makes duration stop being a constraint: only
one chapter is ever submitted at a time. Proven on a 2 h 35 conference.

Boundaries come from a similarity drop between consecutive slices (text
tiling), not from a fixed length, so a chapter ends where the subject changes.
"""
from __future__ import annotations

import math

from geshtu import engines
from geshtu.models import Chapter

MAX_SLICES = 8          # ~16 min: beyond this a cut is forced
MIN_SLICES = 2          # ~4 min: avoids stub chapters


class Chaptering:
    name = "chapter"
    requires = ("segments",)
    provides = ("chapters",)

    def run(self, session, cfg, report) -> None:
        segs = session.segments
        if not segs:
            report("nothing to chapter")
            return
        if len(segs) <= MIN_SLICES:
            session.chapters = [Chapter(segs[0].start, segs[-1].end, "", "")]
            return

        vectors = engines.embed(cfg.engine("embed"), [s.text for s in segs])
        scores = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        mean = sum(scores) / len(scores)
        spread = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        # Calibrated rather than guessed: a fixed cosine threshold means
        # nothing across different content.
        threshold = mean - 0.6 * spread
        report("%d slices, similarity threshold %.3f" % (len(segs), threshold))

        groups = _split([list(range(len(segs)))], scores, threshold)
        session.chapters = [
            Chapter(segs[g[0]].start, segs[g[-1]].end, "", "") for g in groups]
        report("%d chapters" % len(session.chapters))


def _split(groups, scores, threshold):
    out = []
    for g in groups:
        out.extend(_one(g, scores, threshold))
    return out


def _one(idx, scores, threshold):
    if len(idx) <= MAX_SLICES:
        return [idx]
    # Force a cut at the deepest similarity trough inside the group, then
    # recurse: this keeps long monologues from becoming one giant chapter.
    inner = [(scores[i], i) for i in idx[:-1]]
    _, at = min(inner)
    left = [i for i in idx if i <= at]
    right = [i for i in idx if i > at]
    if not left or not right:
        return [idx]
    return _one(left, scores, threshold) + _one(right, scores, threshold)


def _cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


PLUGIN = Chaptering
