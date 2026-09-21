# geshtu

A meeting assistant that runs entirely on your own machine. It records an
audio source, works out **who spoke when**, transcribes it and summarises it
on a local **NPU** and **GPU** through
[OpenVINO](https://github.com/openvinotoolkit/model_server). Nothing leaves
the box.

The Sumerian *ĝeštug* means both *ear* and *understanding*: hearing is where
comprehension happens. Enki is praised as *ĝeštug daĝal*, "of wide ear".

> **Status: early but working end to end.** Proven on real recordings up to
> 2 h 35. The packaging and the plugin API are young and will move.

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
every model as a configurable engine, each with its own endpoint and its own
device — and the model can be changed while it runs.

## What it does

```
 source ─► capture ─► mix ─► diarise ─► transcribe ─► chapter ─► summarise ─► sinks
   │                            NPU         NPU          GPU         GPU        │
   │                                                                            │
 PipeWire, a file, a joiner                          Markdown, Matrix, webhook
```

**Records one application, not the whole desktop.** The system output mixes
everything that plays and a microphone picks up the room. geshtu derives a
single stream into a private sink and leaves it audible, so you can record one
browser tab while another plays something else.

**Diarisation comes first, and that ordering matters.** Transcript slices are
cut on silence and run to minutes; speech turns are cut on changes of voice
and an audience question lasts seconds. Transcribing first freezes a grid in
which a three-second interjection cannot exist — measured: three speakers
found, one single label in the whole note. Diarising first hands its
boundaries to the transcription.

**Duration stops being a constraint.** The transcript is cut by topic and only
one chapter is ever submitted to the model.

**The full transcript is always kept**, folded at the end of the note, with
speaker labels. A summary cannot be checked against itself.

## Measured on an Intel Core Ultra X7 358H (Panther Lake)

| | | |
|---|---|---|
| transcription | NPU | **40× realtime** |
| transcription latency, 15 s chunk | NPU | 0.5 s — live would be ~4% of it |
| diarisation | NPU | **230× realtime** — 2 h 35 in 40 s |
| embeddings, 14 slices | GPU | **0.5 s**, against 46.9 s on the CPU |
| generation | GPU | 25 tokens/s (NPU: 21) |
| prompt ceiling | NPU | **8192 tokens, hard** |
| prompt ceiling | GPU | 21,043 accepted, none found |

**Each accelerator has a job, and the assignment is measured rather than
assumed.** Two results contradicted the obvious choice:

- The **GPU is the worst at segmentation** — 106 ms against 16 ms on the NPU
  and 45 ms on the CPU. The model is small and launch overhead dominates.
- The **NPU refuses the embeddings model** (`VPU.NCE.Reduce` cannot be
  legalised: unbounded dynamic shapes) while accepting both diarisation
  models, whose shapes are fixed. Same chip, opposite answers, so the only way
  to know is to try.

The NPU prompt ceiling is a compiler constant, not a memory limit: 8192
compiles in 75 s, 9216 is refused in 15 s, and a model a gigabyte smaller
fails at exactly the same point. That is why the language model sits on the
GPU — an hour of speech is 10,000–15,000 tokens and would never fit.

## Install

Python 3.11+, `ffmpeg`, and for PipeWire capture `pw-record`, `pw-link`,
`pw-cli`, `pw-dump`. Diarisation needs `openvino`, `numpy` and
`kaldi-native-fbank`, plus the two converted models.

```sh
pip install -e .
geshtud &                      # or a systemd --user unit
```

⚠️ **The NPU vanishes silently without level-zero on `LD_LIBRARY_PATH`.** The
plugin is loaded by `dlopen` and OpenVINO drops the device with no message:
`available_devices` simply returns `['CPU', 'GPU']` and everything falls back
to the CPU while appearing to work.

## Use

```sh
geshtu targets                 # what can be recorded right now
geshtu start node:163          # one application stream
geshtu status
geshtu stop --lang en
geshtu start file:talk.mp4     # the same chain on an existing recording
```

There is also a window (`geshtu-gui`) and a tray icon (`geshtu-tray`), both
thin clients: neither holds any state, both ask the daemon every three
seconds, so a recording started from the keyboard shows up in both.

An application target **only exists while it is playing**, and each stream is
its own target — a browser lists one entry per tab, marked ▶ when it is making
sound and ‖ when it is not. Start the playback first, then list the targets.

## Engines

```toml
[engines.asr]                                   # speech to text
endpoint = "http://localhost:8094/v3/audio/transcriptions"
model    = "whisper"
device   = "NPU"

[engines.llm]                                   # summaries
endpoint   = "http://localhost:8092/v3/chat/completions"
model      = "qwen3-8b"
device     = "GPU"
max_prompt = 0                                  # 0 = no known ceiling

[engines.embed]                                 # topic boundaries
endpoint = "http://localhost:8096/v3/embeddings"
model    = "embeddings"
device   = "GPU"

[diarisation]
device    = "NPU"
speakers  = 4       # when you know: a count always beats a threshold
threshold = 0.9     # higher merges speakers, lower splits one voice in two
```

⚠️ `device` **describes** where an endpoint runs; it does not move anything.
Which accelerator serves a model is decided when the model server starts.

⚠️ **The clustering threshold is sharp, and it is worth setting.** Measured on
a 2 h 35 conference: 0.5 gave 145 speakers, 0.7 gave 43, 0.9 gave 4, 1.1 gave
1. And a wrong value does not only spoil attribution — every false change of
voice becomes its own slice, so that run made 808 transcription calls where
the corrected one made 170.

```sh
geshtu models                  # what each endpoint serves, and its state
geshtu model llm mistral-7b    # switch, now, without a restart
geshtu models add hf:OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov
```

## Plugins

| kind | does | examples |
|---|---|---|
| **source** | produces audio | PipeWire, file, **a joiner that dials into a Teams or Meet call** |
| **stage** | transforms the session | diarise, transcribe, chapter, summarise |
| **sink** | delivers the result | Markdown, Matrix, webhook, wiki |

A stage never calls another stage: it reads and writes the session and stops.
That is what lets you reorder the pipeline, swap a model, or drop a step —
and the session is saved after each one, so a failure in summarising never
costs the transcription.

A source does not have to tap something already playing; it may **make** it:

```sh
geshtu start meet:https://meet.google.com/abc-defg-hij
```

Discovery is by entry point, and a directory of loose `.py` files is scanned
too — drop one in `~/.config/geshtu/stages/` to try an idea. See
[docs/plugins.md](docs/plugins.md) and [docs/models.md](docs/models.md).

## Design rules

**No runtime dependencies in the core.** Compute lives behind HTTP in a model
server. The one exception is diarisation, which drives two 3 MB and 13 MB
models directly: putting them behind a server would cost more plumbing than
calling OpenVINO.

**The daemon holds the state.** Clients are thin. Nothing recomputes status
from files on disk, because when that computation broke, every interface went
blank at once while the recording was perfectly fine.

**Every warning in the code is a measurement.** They are there because
something once produced plausible garbage in silence: a recorder falling back
to the default sink when its target was missing, a hardcoded list of track
kinds dropping a whole track at mix time, a speaker numbered zero treated as
no speaker at all.

## Licence

MIT.
