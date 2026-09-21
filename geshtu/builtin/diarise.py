"""Who spoke when, and attaching that to what was said.

Two models do the work and sherpa-onnx drives them: one cuts the audio into
speech turns, the other gives each turn a voice print which is then
clustered. This stage only runs the tool and folds the answer back into the
transcript.

⚠️ IT RUNS ON THE CPU, and that is not a preference. sherpa-onnx goes through
onnxruntime, whose nixpkgs build carries no OpenVINO provider: neither the
NPU nor the GPU is reachable from it. An OpenVINO conversion of the same two
models exists (FluidInference/speaker-diarization-ov) with FIXED input shapes
-- which is precisely what the NPU needs, and precisely what the embeddings
model lacked -- but it ships weights only; the sliding window, the
binarisation and the clustering would all have to be rewritten. Measure the
CPU first, port when the measurement says it is worth it.

Measured on a Core Ultra X7 358H, 67 s of speech:

    1 thread   8.34 s   RTF 0.125
    4 threads  4.09 s   RTF 0.061      <- chosen
    8 threads  3.62 s   RTF 0.054

Four threads halve it; eight add 11% for twice the cores. An hour of meeting
costs about 3 min 40.

⚠️ THE NUMBER OF SPEAKERS IS NOT KNOWN IN ADVANCE, so clustering is driven by
a threshold rather than a count. Too low and one person becomes three; too
high and a meeting collapses into one speaker. 0.8 is the upstream default
and the place to start when attribution looks wrong.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

LIGNE = re.compile(r"^\s*([\d.]+)\s*--\s*([\d.]+)\s+(\S+)\s*$")


class Diarise:
    name = "diarise"
    requires = ("segments",)
    provides = ("speakers",)

    def run(self, session, cfg, report) -> None:
        models = os.environ.get("GESHTU_DIARISATION")
        binary = shutil.which("sherpa-onnx-offline-speaker-diarization")
        if not models or not binary:
            # ⚠️ ABSENT IS NOT BROKEN. A pip install has neither the models
            # nor the tool, and a meeting summary without speaker labels is
            # still a meeting summary. Say it once and move on.
            report("diarisation unavailable (no models or no sherpa-onnx)")
            return
        if not session.mixed or not session.mixed.exists():
            report("nothing to diarise")
            return

        opts = cfg.raw.get("diarisation", {})
        cmd = [
            binary,
            "--segmentation.pyannote-model=%s/segmentation.onnx" % models,
            "--embedding.model=%s/embedding.onnx" % models,
            "--segmentation.num-threads=%d" % opts.get("threads", 4),
            "--embedding.num-threads=%d" % opts.get("threads", 4),
            "--clustering.cluster-threshold=%s" % opts.get("threshold", 0.8),
            str(session.mixed),
        ]
        report("diarising %s" % session.mixed.name)
        out = subprocess.run(cmd, capture_output=True, text=True)
        tours = []
        for ligne in out.stdout.splitlines():
            m = LIGNE.match(ligne)
            if m:
                tours.append((float(m.group(1)), float(m.group(2)), m.group(3)))
        if not tours:
            report("diarisation produced nothing:\n%s" % out.stderr[-400:])
            return

        noms = {}
        for _, _, qui in tours:
            noms.setdefault(qui, "Speaker %d" % (len(noms) + 1))
        report("%d turns, %d speakers" % (len(tours), len(noms)))

        # ⚠️ ATTRIBUTION BY LARGEST OVERLAP, not by whoever starts first. A
        # transcript slice is cut on silence and a speech turn is cut on voice
        # activity: the two grids do not line up, and a slice routinely begins
        # inside the previous person's tail.
        for seg in session.segments:
            best, meilleur = None, 0.0
            for debut, fin, qui in tours:
                part = min(seg.end, fin) - max(seg.start, debut)
                if part > meilleur:
                    best, meilleur = qui, part
            seg.speaker = noms.get(best) if best else None


PLUGIN = Diarise
