"""Treat an existing media file as a session.

The same chain, applied to a downloaded video or a recording made elsewhere:
extract the audio, and from there nothing differs. Proven on a 2 h 35
conference and on a Japanese demo video -- the ASR detects the language, which
is precisely why nothing here forces one.
"""
from __future__ import annotations

import pathlib
import subprocess

from geshtu import plugins
from geshtu.models import Track


class FileSource:
    name = "file"

    def targets(self) -> list[plugins.Target]:
        # A file is named by the caller, not discovered. Nothing to list.
        return []

    def start(self, session, target: plugins.Target, index: int) -> Track:
        # The track kind matches the source name, which is how the daemon
        # knows whose stop() to call when the session ends.
        src = pathlib.Path(target.key).expanduser()
        if not src.is_file():
            raise RuntimeError("no such file: %s" % src)
        wav = session.root / ("file-%03d.wav" % index)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                        str(wav)], check=True)
        return Track(kind="file", source=src.name, segments=[wav])

    def stop(self, session) -> None:
        return None


PLUGIN = FileSource
