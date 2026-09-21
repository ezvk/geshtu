# geshtu

A meeting assistant that runs entirely on your own machine: it records an
audio source, transcribes it and summarises it on a local **NPU** or **GPU**,
through [OpenVINO](https://github.com/openvinotoolkit/model_server). Nothing
leaves the box, and the GPU stays free while it works.

The Sumerian *ĝeštug* means both *ear* and *understanding*: hearing is where
comprehension happens. Enki is praised as *ĝeštug daĝal*, "of wide ear".

> **Status: early.** The pipeline is proven end to end on real recordings.
> The packaging and the plugin API are young and will move.

---

## Why build it

Every open-source meeting assistant **embeds** its speech recognition and only
lets you configure the language model:

| | remote STT | remote LLM |
|---|---|---|
| Meetily | no | yes |
| Anarlog | no (Deepgram only) | yes |
| Vibe | no — issue open 20 months | yes |
| Scriberr | no | yes |
| Handy | no — PR closed unmerged | yes |

If you already run local inference, that is exactly backwards. geshtu treats
speech recognition, embeddings and generation as **three configurable
engines**, each with its own endpoint, its own model and its own device — and
the model can be changed while it runs, because which model is loaded changes
under you.

## What it does

```
 source ─► capture ─► mix ─► transcribe ─► chapter ─► summarise ─► sinks
   │                            NPU          CPU         NPU         │
   │                                                                 │
 PipeWire, a file                              Markdown, Matrix, webhook
```

**Records one application, not the whole desktop.** The system output mixes
everything that plays and a microphone picks up the room. geshtu derives a
single stream into a private sink and leaves it audible — so you can record
one browser tab while another plays something else.

**Duration stops being a constraint.** The transcript is cut by topic and only
one chapter is ever submitted to the model.

**The full transcript is always kept**, folded at the end of the note. A
summary cannot be checked against itself, and that is exactly what you need
the day the model invents something.

## Measured on an Intel Core Ultra X7 358H (Panther Lake)

| | |
|---|---|
| transcription, NPU | **40× realtime** — an hour of audio in ~90 s |
| generation, NPU | ~20 tokens/s |
| prompt ceiling, NPU | **8192 tokens, hard** |
| one-hour meeting, end to end | ~5 minutes, GPU idle |

**The ceiling is real and cannot be raised.** Found by bisection: 8192
compiles in 75 s, 9216 is refused in 15 s, and so is 16384. A model a gigabyte
smaller fails at exactly the same point, so it is a constant of the compiler,
not a memory limit — not a BIOS setting, not a configuration. An hour of
speech is 10,000–15,000 tokens, which is why chaptering is the architecture
and not an optimisation.

## Engines

Three roles, each pointed wherever you like. Moving a model from the NPU to
the GPU is one line; nothing in the pipeline knows the difference.

```toml
[engines.asr]                                   # speech to text
endpoint = "http://localhost:8094/v3/audio/transcriptions"
model    = "whisper"
device   = "NPU"

[engines.llm]                                   # summaries
endpoint   = "http://localhost:8092/v3/chat/completions"
model      = "qwen3-8b"
device     = "NPU"
max_prompt = 8192

[engines.embed]                                 # topic boundaries
endpoint = "http://localhost:8096/v3/embeddings"
model    = "embeddings"
device   = "CPU"
```

Two accelerators, three roles: transcription and generation are sequential, so
they share the NPU without contending, and the GPU is never touched — it stays
available for whatever else you are doing.

```sh
geshtu models                  # what each endpoint serves, and its state
geshtu model llm mistral-7b    # switch, now, without a restart
```

The model name is the name the **server** serves, which is rarely the name of
the weights. A name that matches nothing fails at the first request — an hour
after the recording.

## Install

Python 3.11+, `ffmpeg`, and for PipeWire capture `pw-record`, `pw-link`,
`pw-cli`, `pw-dump`.

```sh
pip install -e .
geshtud &                      # or a systemd --user unit
```

Any endpoint speaking the OpenAI shape will do. The reference setup is
OpenVINO Model Server with the three models above.

## Use

```sh
geshtu targets                 # what can be recorded right now
geshtu start node:163 alsa_input...   # one or more targets
geshtu status
geshtu stop --lang en          # transcribe, chapter, summarise
geshtu start file:talk.mp4     # the same chain on an existing recording
```

An application target **only exists while it is playing**, and each stream is
its own target — a browser lists one entry per tab, named after the tab.
Start the playback first, then list the targets.

## Plugins

Three kinds, because a meeting assistant has three moving parts:

| kind | does | examples |
|---|---|---|
| **source** | produces audio | PipeWire, file, **a joiner that dials into a Teams or Meet call** |
| **stage** | transforms the session | transcribe, diarise, chapter, summarise |
| **sink** | delivers the result | Markdown, Matrix, webhook, wiki |

A stage never calls another stage: it reads and writes the session and stops.
That is what lets you reorder the pipeline, swap a model, or drop a step
without touching anything else — and the session is saved after each one, so a
failure in summarising never costs the transcription.

A source does not have to tap something already playing — it may **make** it.
That is how a meeting joiner fits: handed a URL, it opens the call and the
rest of the chain is unchanged.

```sh
geshtu start meet:https://meet.google.com/abc-defg-hij
```

Discovery is by entry point, so a package installed alongside is picked up
with no registration. A directory of loose `.py` files is scanned too — drop
one in `~/.config/geshtu/stages/` to try an idea. See
[docs/plugins.md](docs/plugins.md).

## Design rules

**No runtime dependencies.** Compute lives behind HTTP in a model server, so
there is no torch and no CUDA in the core — only the standard library.

**The daemon holds the state.** Clients are thin: a CLI, a window, a tray
icon. Nothing recomputes status from files on disk, because when that
computation broke, every interface went blank at once while the recording was
perfectly fine.

**Every warning in the code is a measurement.** They are there because
something once produced plausible garbage in silence — a recorder falling back
to the default sink when its target was missing, a hardcoded list of track
kinds quietly dropping a whole track at mix time, an ASR hallucinating a
recipe over an empty file.

## Licence

MIT.
