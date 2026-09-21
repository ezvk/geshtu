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
device = "NPU"
# Measured ceiling on Intel NPU (Panther Lake, 2026-09-19): compilation is
# refused above 8192 whatever the model -- Qwen3-8B and Mistral-7B fail at the
# same 9216. It is a compiler constant, not a memory limit, so it cannot be
# raised. Everything above this is chaptered instead.
max_prompt = 8192

[engines.embed]
endpoint = "http://localhost:8096/v3/embeddings"
model = "bge-m3"
device = "CPU"

[pipeline]
stages = ["transcribe", "chapter", "summarise"]
sinks = ["markdown"]
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
