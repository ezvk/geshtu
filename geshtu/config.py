"""Configuration, and the engine table that makes NPU/GPU placement a setting.

The whole point of separating engines from stages is that WHERE a model runs
is a deployment fact, not a code fact. A stage asks for the engine named
"asr"; whether that is an NPU at 40x realtime, an iGPU, or a remote box is
decided here.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import tomllib

DEFAULT = """
[paths]
# sessions = "~/.local/share/geshtu"

[capture]
rate = 16000
channels = 1

[engines.asr]
endpoint = "http://localhost:8094/v3/audio/transcriptions"
model = "whisper"
device = "NPU"

[engines.llm]
endpoint = "http://localhost:8092/v3/chat/completions"
model = "qwen3-8b"
# ⚠️ `device` DESCRIBES where this endpoint runs; it does not move anything.
# Which accelerator serves a model is decided when the model server is
# started. This field exists so the pipeline can reason about the endpoint's
# limits, and so `geshtu models` can tell you what you are actually talking
# to -- not so a line here can relocate a model.
device = "GPU"
# The prompt ceiling of this endpoint, in tokens. 0 means no known limit.
#
# ⚠️ IT IS A PROPERTY OF THE ACCELERATOR, NOT OF THE MODEL. Measured on an
# Intel NPU (2026-09-19) by bisection: 8192 compiles in 75 s, 9216 is refused
# in 15 s, and a model a gigabyte smaller fails at exactly the same point --
# a compiler constant, not a memory limit. The same weights on the GPU
# accepted 21 043 tokens and answered correctly (2026-09-21).
#
# An hour of speech is 10,000-15,000 tokens, so this number decides whether a
# meeting is summarised whole or in chapters.
max_prompt = 0

[engines.embed]
endpoint = "http://localhost:8096/v3/embeddings"
# The name the SERVER serves, which is rarely the name of the weights.
# `geshtu models` prints what each endpoint actually offers; a name that
# matches nothing fails at the first request, an hour after the recording.
model = "embeddings"
device = "CPU"

[pipeline]
stages = ["transcribe", "chapter", "summarise"]
sinks = ["markdown"]

# Dropping a model in a folder and having it served. The server must be
# started with --config_path <config> --file_system_poll_wait_seconds N,
# otherwise it never re-reads what geshtu writes here.
# [repository]
# ovms = "ovms"
# models = "/var/lib/geshtu/models"
# config = "/var/lib/geshtu/ovms.json"
"""


@dataclasses.dataclass
class Engine:
    name: str
    endpoint: str
    model: str
    device: str = "CPU"
    max_prompt: int | None = None
    timeout: float = 600.0


@dataclasses.dataclass
class Config:
    sessions: pathlib.Path
    rate: int
    channels: int
    engines: dict[str, Engine]
    stages: list[str]
    sinks: list[str]
    raw: dict

    def engine(self, name: str) -> Engine:
        try:
            return self.engines[name]
        except KeyError:
            raise SystemExit(
                "no engine named %r; declare [engines.%s] in the config" % (name, name))

    def with_models(self, chosen: dict) -> "Config":
        """A copy of this config with some engines pointed at another model.

        ⚠️ WHICH MODEL IS RUNNING CHANGES UNDER YOU. A model server is
        reconfigured, a model is swapped for a smaller one to free the
        accelerator, a name gains a version suffix. Pinning the name in a file
        that only a rebuild can change turns an ordinary Tuesday into an
        outage, so the choice is runtime state, kept beside the sessions.
        """
        if not chosen:
            return self
        engines = {}
        for name, eng in self.engines.items():
            engines[name] = dataclasses.replace(eng, model=chosen[name]) \
                if chosen.get(name) else eng
        return dataclasses.replace(self, engines=engines)


def path() -> pathlib.Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (pathlib.Path.home() / ".config")
    return pathlib.Path(base) / "geshtu" / "geshtu.toml"


def load(explicit: pathlib.Path | None = None) -> Config:
    p = explicit or path()
    raw = tomllib.loads(DEFAULT)
    if p.is_file():
        merge(raw, tomllib.loads(p.read_text()))
    data = pathlib.Path(
        os.environ.get("XDG_DATA_HOME") or (pathlib.Path.home() / ".local/share"))
    sessions = raw.get("paths", {}).get("sessions")
    return Config(
        sessions=pathlib.Path(sessions).expanduser() if sessions else data / "geshtu",
        rate=raw["capture"]["rate"],
        channels=raw["capture"]["channels"],
        engines={k: Engine(name=k, **v) for k, v in raw.get("engines", {}).items()},
        stages=raw["pipeline"]["stages"],
        sinks=raw["pipeline"]["sinks"],
        raw=raw,
    )


def merge(base: dict, over: dict) -> None:
    """Deep merge, so a user config may set one key of one engine without
    having to restate the whole table."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
