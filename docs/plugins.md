# Writing a plugin

Two ways to be found:

1. **An entry point** in your own package — the normal way to distribute one.
2. **A file dropped in `~/.config/geshtu/{sources,stages,sinks}/`** exporting
   a `PLUGIN` symbol. No packaging, no install; it overrides a builtin of the
   same name. This is the way to try an idea.

A plugin that raises while loading is reported and skipped: a broken
third-party plugin must never take the daemon down.

## A stage

```python
class Redact:
    name = "redact"
    requires = ("segments",)
    provides = ("segments",)

    def run(self, session, cfg, report):
        for seg in session.segments:
            seg.text = scrub(seg.text)
        report("redacted %d segments" % len(session.segments))

PLUGIN = Redact
```

Then put it in the chain:

```toml
[pipeline]
stages = ["transcribe", "redact", "chapter", "summarise"]
```

`report` takes a short status string and is how a long stage stays visible to
whoever is watching. Use it every few seconds, not once at the end.

The session is saved after every stage, so a failure in a later one never
costs the transcription. Audio is the only thing here that cannot be
recomputed, and transcription is the only thing that takes real time.

## A sink

```python
class Webhook:
    name = "webhook"

    def deliver(self, session, cfg, report):
        post(cfg.raw["webhook"]["url"], session.summary)

PLUGIN = Webhook
```

A sink failing is **not** a pipeline failure: the transcript and the summary
already exist, and an outage on your chat server must not make the run look
lost.

## A source

```python
class Rtsp:
    name = "rtsp"

    def targets(self):          # what can be recorded right now
        return [Target(key=url, label="Meeting room", kind="app")]

    def start(self, session, target, index):
        ...                     # must return immediately, with a Track
    def stop(self, session):
        ...                     # must WAIT for files to be closed
```

`Target.key` must survive a restart where that makes sense. A device does:
name it, never use a numeric PipeWire id, which is reassigned on every boot.
An application stream does not — it dies with its process, so precision wins
over persistence there.

## A source that has nothing to enumerate

Some sources do not tap what is already playing: they **make** it. A file is
named, not discovered. A meeting joiner is handed a URL, opens the call, and
produces audio that did not exist a second earlier.

Those are addressed as `<source name>:<whatever the source understands>`, and
`targets()` returns an empty list:

```sh
geshtu start file:~/Downloads/talk.mp4
geshtu start meet:https://meet.google.com/abc-defg-hij
```

The prefix is matched against installed source names, so PipeWire's own
`node:163` keys are not mistaken for a source called "node".

⚠️ **A joiner does not have to carry its own audio stack.** The cheap and
robust shape is to launch a browser into the call with a distinctive PipeWire
node name, then reuse the derivation the PipeWire source already does: join,
tap that node, and the rest of the chain is unchanged. Writing a second audio
path would double the number of places a recording can silently end up empty.

⚠️ **And a joiner is a participant.** It shows up in the attendee list under
whatever name you give it, and in most jurisdictions recording a conversation
requires telling the people in it. Name it so that it is obvious what it is.

`stop()` must wait for the writers to exit. A WAV read before its RIFF header
is finalised makes the ASR return an intermittent HTTP 400 — intermittent
because it depends on how much the writer had flushed.
