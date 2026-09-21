"""The data that flows through a session.

These types are the contract between plugins. A stage receives a Session,
mutates it, and hands it on; it never talks to another stage directly. That is
what makes the pipeline reorderable and a plugin replaceable.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import time


@dataclasses.dataclass
class Track:
    """One captured audio stream, written to disk as it arrives.

    Tracks are SIMULTANEOUS, not sequential. Mixing that up is the single most
    expensive mistake in this domain: concatenating two parallel tracks doubles
    the duration, halves the level, and shifts every timestamp. Anything that
    walks a list of audio files must know which of the two it is holding.
    """

    kind: str                      # "mic" | "output" | "app"
    source: str                    # human-readable, e.g. "Firefox"
    segments: list[pathlib.Path] = dataclasses.field(default_factory=list)
    level_db: float | None = None  # measured once capture stops

    @property
    def silent(self) -> bool:
        """A track nobody spoke into. Worth saying out loud at stop time: it
        is the only moment the user can still fix a wrong source choice."""
        return self.level_db is not None and self.level_db < -70


@dataclasses.dataclass
class Segment:
    """A transcribed slice. Timestamps come from US, not from the ASR engine.

    OpenVINO Model Server ignores `response_format`: verbose_json, srt and vtt
    all return a bare {"text": ...}. There is nothing to recover from the
    engine, so the pipeline slices on silence and remembers where it cut.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclasses.dataclass
class Turn:
    """A stretch of one voice. Produced before transcription, because the two
    grids must agree: slices cut only on silence cannot carry a three-second
    interjection inside a two-minute stretch of someone else."""

    start: float
    end: float
    speaker: str


@dataclasses.dataclass
class Chapter:
    start: float
    end: float
    title: str
    body: str = ""


@dataclasses.dataclass
class Session:
    """Everything known about one recording, from first sample to final note."""

    id: str
    root: pathlib.Path
    language: str = "en"
    started: float = dataclasses.field(default_factory=time.time)
    ended: float | None = None
    tracks: list[Track] = dataclasses.field(default_factory=list)
    mixed: pathlib.Path | None = None
    turns: list[Turn] = dataclasses.field(default_factory=list)
    segments: list[Segment] = dataclasses.field(default_factory=list)
    chapters: list[Chapter] = dataclasses.field(default_factory=list)
    summary: str = ""
    title: str = ""

    @property
    def duration(self) -> float:
        """Longest track, never the sum. See the warning on Track."""
        return max((seg_total(t) for t in self.tracks), default=0.0)

    # -- persistence ----------------------------------------------------
    #
    # A session is written to disk after every stage, so a crash during
    # summarisation never costs the transcription that preceded it. The audio
    # is the only thing that cannot be recomputed.

    def save(self) -> None:
        tmp = self.root / "session.json.tmp"
        tmp.write_text(json.dumps(as_dict(self), indent=2, ensure_ascii=False))
        tmp.replace(self.root / "session.json")

    @classmethod
    def load(cls, root: pathlib.Path) -> "Session":
        raw = json.loads((root / "session.json").read_text())
        return from_dict(cls, raw, root)


def seg_total(track: Track) -> float:
    from geshtu import audio
    return sum(audio.duration(p) for p in track.segments)


def as_dict(obj):
    """⚠️ IT MUST RECURSE INTO PLAIN DICTS, not only into dataclasses.

    `dataclasses.asdict` already flattens nested dataclasses into dicts, so a
    version that only recursed on dataclasses walked straight past them and
    left Path objects in place. json.dumps then raised "Object of type
    PosixPath is not JSON serializable" -- inside save(), which runs after the
    recorders have started. The recording was live, the session was never
    registered, and stop() answered "not recording" while leaving an orphan
    recorder behind. One missing branch, three visible symptoms.
    """
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: as_dict(v) for k, v in obj.items()}
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [as_dict(x) for x in obj]
    return obj


def from_dict(cls, raw: dict, root: pathlib.Path) -> Session:
    s = cls(id=raw["id"], root=root)
    s.language = raw.get("language", "en")
    s.started = raw.get("started", 0.0)
    s.ended = raw.get("ended")
    s.title = raw.get("title", "")
    s.summary = raw.get("summary", "")
    s.mixed = pathlib.Path(raw["mixed"]) if raw.get("mixed") else None
    s.tracks = [Track(kind=t["kind"], source=t["source"],
                      segments=[pathlib.Path(p) for p in t["segments"]],
                      level_db=t.get("level_db")) for t in raw.get("tracks", [])]
    s.turns = [Turn(**x) for x in raw.get("turns", [])]
    s.segments = [Segment(**x) for x in raw.get("segments", [])]
    s.chapters = [Chapter(**x) for x in raw.get("chapters", [])]
    return s
