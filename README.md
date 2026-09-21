# geshtu

Self-hosted meeting intelligence. Records an audio source, transcribes it and
summarises it **on your own machine**, on an NPU or a GPU — nothing leaves the
box.

The Sumerian *ĝeštug* means both *ear* and *understanding*: hearing is where
comprehension happens. Enki is praised as *ĝeštug daĝal*, "of wide ear".

> Status: early. The pipeline is proven end to end on real recordings; the
> packaging and the plugin API are young and will move.

---

## Why this exists

Every open-source meeting assistant **embeds** its speech recognition and only
lets you configure the language model:

| | remote STT | remote LLM |
|---|---|---|
| Meetily | no | yes |
| Anarlog | no (Deepgram only) | yes |
| Vibe | no — issue open 20 months | yes |
| Scriberr | no | yes |
| Handy | no — PR closed unmerged | yes |

If you already run local inference, that is exactly backwards: you want to
point **both** at your own endpoints. geshtu treats speech recognition,
embeddings and generation as three configurable engines, each with its own
device.

## What it does

```
source ──► capture ──► mix ──► transcribe ──► chapter ──► summarise ──► sinks
```

- **Capture one application, not the whole desktop.** Recording the system
  output mixes everything that plays; a microphone picks up the room. geshtu
  derives a single stream into a private sink, leaving it audible.
- **Duration is not a constraint.** The transcript is split by topic and only
  one chapter is ever submitted to the model.
- **The full transcript is always kept**, folded at the end of the note. A
  summary cannot be checked against itself.

## Measured, on an Intel Core Ultra X7 358H (Panther Lake NPU)

| | |
|---|---|
| transcription | **40× realtime** — one hour of audio in ~90 s |
| prompt ceiling | **8192 tokens, hard** — see below |
| generation | ~20 tokens/s |
| one-hour meeting, end to end | ~5 minutes, GPU idle |

**The ceiling is real and cannot be raised.** Found by bisection: 8192
compiles in 75 s, 9216 is refused in 15 s, and so is 16384. A model a gigabyte
smaller fails at exactly the same point, so it is a compiler constant, not a
memory limit — not a BIOS setting, not a configuration. One hour of speech is
10,000–15,000 tokens, which is why chaptering is the architecture rather than
an optimisation.

## Install

Requires Python 3.11+, `ffmpeg`, and — for PipeWire capture — `pw-record`,
`pw-link`, `pw-cli`, `pw-dump`.

```sh
pip install -e .
geshtu daemon &          # or run it under systemd --user
```

Inference endpoints are expected to speak the OpenAI shape. The reference
setup is [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
with three models; see `docs/engines.md`.

## Use

```sh
geshtu targets                      # what can be recorded right now
geshtu start Firefox alsa_input...  # one or more targets
geshtu status
geshtu stop --lang en               # transcribe, chapter, summarise
```

An application target **only exists while it is playing**. Start the playback
first, then list the targets.

## Configuration

`~/.config/geshtu/geshtu.toml`. Everything has a default; override only what
you need.

```toml
[engines.asr]
endpoint = "http://localhost:8094/v3/audio/transcriptions"
model    = "whisper"
device   = "NPU"

[engines.llm]
endpoint   = "http://localhost:8092/v3/chat/completions"
model      = "qwen3-8b"
device     = "NPU"
max_prompt = 8192

[pipeline]
stages = ["transcribe", "chapter", "summarise"]
sinks  = ["markdown"]
```

Where a model runs is a deployment fact, not a code fact. Move `device` to
`GPU` and nothing else changes.

## Plugins

Three kinds, because a meeting assistant has three moving parts:

| kind | does | examples |
|---|---|---|
| **source** | produces audio | PipeWire, file, RTSP |
| **stage** | transforms the session | transcribe, diarise, chapter, summarise |
| **sink** | delivers the result | Markdown, Matrix, webhook, wiki |

A stage never calls another stage: it reads and writes the session and stops.
That is what lets you reorder the pipeline, swap a model, or drop a step
without touching anything else.

Discovery is by entry point, so a package installed alongside is picked up
with no registration. A directory of loose `.py` files is scanned too — see
`docs/plugins.md`.

## Licence

MIT.
