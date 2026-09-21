"""The plugin contracts, and how they are found.

Three kinds, because a meeting assistant has exactly three moving parts:

    Source   produces audio       (a PipeWire stream, a file, an RTSP feed)
    Stage    transforms a Session (transcribe, diarise, chapter, summarise)
    Sink     delivers the result  (Markdown, Matrix, a webhook, a wiki)

A Stage never calls another Stage. It reads and writes the Session and stops.
That is the whole reason the pipeline can be reordered, a stage swapped for a
better model, or a step removed, without touching anything else.

Discovery is by entry point, so an unrelated package installed alongside is
picked up with no registration and no edit to this project. A directory of
loose .py files is also scanned, which is what makes a one-file plugin
possible without packaging it.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import os
import pathlib
import typing

if typing.TYPE_CHECKING:
    from geshtu.config import Config
    from geshtu.models import Session, Track

GROUPS = ("geshtu.sources", "geshtu.stages", "geshtu.sinks")


@dataclasses.dataclass
class Target:
    """Something a Source can record: a microphone, a sink monitor, an
    application stream. `key` is what survives a restart -- never a numeric id,
    which PipeWire reassigns on every boot."""

    key: str
    label: str
    kind: str                     # "mic" | "output" | "app"
    detail: str = ""


class Source(typing.Protocol):
    name: str

    def targets(self) -> list[Target]:
        """What can be recorded right now. An application stream only exists
        while it is playing -- an empty list is a normal answer."""

    def start(self, session: "Session", target: Target, index: int) -> "Track":
        """Begin writing a new segment. Must return immediately."""

    def stop(self, session: "Session") -> None:
        """Stop every recorder this source owns and WAIT for the files to be
        closed, so the RIFF headers are finalised before anything reads them."""


class Stage(typing.Protocol):
    name: str
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def run(self, session: "Session", cfg: "Config", report) -> None:
        """Mutate the session in place. `report` takes a short status string;
        it is how a long stage stays visible to whoever is watching."""


class Sink(typing.Protocol):
    name: str

    def deliver(self, session: "Session", cfg: "Config", report) -> None:
        ...


def _from_entry_points(group: str) -> dict[str, type]:
    found: dict[str, type] = {}
    for ep in importlib.metadata.entry_points(group=group):
        try:
            found[ep.name] = ep.load()
        except Exception as exc:                        # noqa: BLE001
            # ⚠️ A broken third-party plugin must not take the daemon down
            # with it. It is reported and skipped.
            print("geshtu: plugin %r in %s failed to load: %s" % (ep.name, group, exc))
    return found


def _from_directory(group: str) -> dict[str, type]:
    base = pathlib.Path(
        os.environ.get("XDG_CONFIG_HOME") or (pathlib.Path.home() / ".config"))
    folder = base / "geshtu" / group.split(".")[-1]
    found: dict[str, type] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.py")):
        spec = importlib.util.spec_from_file_location("geshtu_local_" + path.stem, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:                        # noqa: BLE001
            print("geshtu: local plugin %s failed: %s" % (path, exc))
            continue
        obj = getattr(module, "PLUGIN", None)
        if obj is not None:
            found[getattr(obj, "name", path.stem)] = obj
    return found


BUILTIN = {
    "geshtu.sources": {"pipewire": ("geshtu.builtin.pipewire", "PipeWireSource"),
                       "file": ("geshtu.builtin.filesource", "FileSource")},
    "geshtu.stages": {"transcribe": ("geshtu.builtin.asr", "Transcribe"),
                      "diarise": ("geshtu.builtin.diarise", "Diarise"),
                      "chapter": ("geshtu.builtin.chapters", "Chaptering"),
                      "summarise": ("geshtu.builtin.summary", "Summarise")},
    "geshtu.sinks": {"markdown": ("geshtu.builtin.markdown", "MarkdownSink")},
}


def _builtin(group: str) -> dict[str, type]:
    """⚠️ SO THAT A PLAIN CHECKOUT RUNS. Entry points only exist once the
    package is installed; without this fallback, `python -m geshtu.cli` from a
    clone finds no sources at all and the daemon looks broken when nothing is.
    """
    import importlib
    found = {}
    for name, (module, attr) in BUILTIN.get(group, {}).items():
        try:
            found[name] = getattr(importlib.import_module(module), attr)
        except Exception as exc:                        # noqa: BLE001
            print("geshtu: builtin %r unavailable: %s" % (name, exc))
    return found


def registry(group: str) -> dict[str, type]:
    """Builtins first, then entry points, then the local directory -- so an
    installed plugin overrides a builtin of the same name, and a file dropped
    in the config directory overrides both while you experiment."""
    found = _builtin(group)
    found.update(_from_entry_points(group))
    found.update(_from_directory(group))
    return found


def sources() -> dict[str, type]:
    return registry("geshtu.sources")


def stages() -> dict[str, type]:
    return registry("geshtu.stages")


def sinks() -> dict[str, type]:
    return registry("geshtu.sinks")
