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

`Target.key` must survive a restart. Never use a numeric PipeWire id: they are
reassigned on every boot.

`stop()` must wait for the writers to exit. A WAV read before its RIFF header
is finalised makes the ASR return an intermittent HTTP 400 — intermittent
because it depends on how much the writer had flushed.
