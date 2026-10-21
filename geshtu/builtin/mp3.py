"""Keep the recording as an MP3 beside its note, then clear the WAV.

ezvk, 2026-10-21: « il faudrait que geshtu sauvegarde le fichier audio » --
« 1 en mp3 fais le menage ». The audio was in fact kept, but where nobody looks
(~/.local/share/geshtu) and three times over: each track, mixed.wav, and the
transcription slices -- 12 GB after a month. Now: one MP3 named like the note
(`<base>.mp3` beside `<base>-transcription.md`), and the session folder keeps
only session.json.

⚠️ IT MUST RUN BEFORE THE MARKDOWN SINK, which links the MP3 only when it
exists: sinks = ["mp3", "markdown"].
"""
from __future__ import annotations

import shutil
import subprocess

from geshtu import audio
from geshtu.builtin.markdown import emplacement

BITRATE = "64k"         # mono speech at 16 kHz: about 28 MB per hour


class Mp3Sink:
    name = "mp3"

    def deliver(self, session, cfg, report) -> None:
        # Already done: the names were changed and the note is being rewritten.
        if session.audio and session.audio.exists():
            return
        # No note, no MP3 beside it -- and the WAV stay, so nothing is lost
        # when the transcription failed and the session is reprocessed later.
        if not session.segments:
            return
        if not audio.ensure_mixed(session, cfg, report):
            return
        out, base = emplacement(session, cfg)
        out.mkdir(parents=True, exist_ok=True)
        mp3 = out / (base + ".mp3")
        tmp = out / (base + ".mp3.part")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(session.mixed), "-c:a", "libmp3lame",
                        "-b:a", BITRATE, "-ac", "1", "-f", "mp3", str(tmp)],
                       check=True)
        # ⚠️ THE WAV ARE DELETED ONLY AFTER THE MP3 HAS BEEN CHECKED: same
        # length as the mix to the second. Past this point it is the only copy.
        attendu, obtenu = audio.duration(session.mixed), audio.duration(tmp)
        if abs(attendu - obtenu) > 1.0:
            tmp.unlink(missing_ok=True)
            report("mp3 is %.0f s for a %.0f s mix -- WAV kept" % (obtenu, attendu))
            return
        tmp.replace(mp3)
        session.audio = mp3
        report("audio kept: %s" % mp3)
        self._menage(session, report)
        session.save()

    def _menage(self, session, report) -> None:
        """The recording's own audio, inside the session folder only.

        ⚠️ NOT EVERY WAV IN THERE: resume-<key>.wav is the read-aloud cache of
        the summary (lecture.py), meant to be kept. Named one by one instead."""
        avant = _taille(session.root)
        shutil.rmtree(session.root / "slices", ignore_errors=True)
        jetes = [p for t in session.tracks for p in t.segments]
        jetes += [session.root / "mixed.wav"]
        jetes += list(session.root.glob("track-*.wav")) + list(session.root.glob("concat-*.txt"))
        jetes += list(session.root.glob(".direct-*.wav"))
        for f in jetes:
            if f.parent == session.root:
                f.unlink(missing_ok=True)
        session.mixed = None
        report("session folder cleared: %d MB freed"
               % ((avant - _taille(session.root)) // 1_000_000))


def _taille(dossier) -> int:
    return sum(f.stat().st_size for f in dossier.rglob("*") if f.is_file())


PLUGIN = Mp3Sink
