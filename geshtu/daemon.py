"""The daemon: it holds the session, the recorders and the work queue.

⚠️ WHY A DAEMON AT ALL. The previous generation of this tool was a plain CLI
invoked by a keyboard shortcut, and it had two structural problems that no
amount of care fixes:

  - nothing could watch a run. Status was recomputed by every client from
    files on disk, so a bug in that computation blanked every UI at once while
    the recording itself was fine;
  - a long stage had no owner. Whoever pressed stop had to stay alive for the
    ten minutes of transcription and summarisation.

A daemon holds the state once, and clients are thin.

The protocol is newline-delimited JSON over a unix socket. No HTTP, no
dependency, and anything that can write a line can drive it:

    printf '{"cmd":"status"}\n' | nc -U $XDG_RUNTIME_DIR/geshtu.sock
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import socketserver
import subprocess
import threading
import time
import traceback

from geshtu import commandes
from geshtu import config
from geshtu import engines
from geshtu import models
from geshtu import pipeline
from geshtu import plugins


def socket_path() -> pathlib.Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return pathlib.Path(base) / "geshtu.sock"


class State:
    """Everything mutable, behind one lock. Small enough to stay obvious."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.session: models.Session | None = None
        # ⚠️ SEPARATE SLOTS FROM `self.session`, ON PURPOSE. A meeting
        # recording and a quick dictation both want the microphone, and
        # PipeWire allows several simultaneous captures of the same node --
        # there is nothing to arbitrate. MacParakeet's own architecture
        # description names this directly: "a reserved dictation slot and a
        # shared meeting/file slot". Dictating over a meeting must not be
        # refused, and must not disturb it.
        self.dictee_session: models.Session | None = None
        self.commande_session: models.Session | None = None
        self.source_name = "pipewire"
        self.busy = ""                       # non-empty while a stage runs
        self.log: list[str] = []
        self.jobs: queue.Queue = queue.Queue()
        self.sources = {n: c() for n, c in plugins.sources().items()}
        self.models = self._load_models()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- model choice ---------------------------------------------------
    #
    # Kept beside the sessions rather than in the config file: the config is
    # what the machine is set up to have, this is what is loaded right now.

    def _models_path(self) -> pathlib.Path:
        return self.cfg.sessions / "engines.json"

    def _load_models(self) -> dict:
        try:
            return json.loads(self._models_path().read_text())
        except (OSError, ValueError):
            return {}

    def active(self) -> config.Config:
        """The config as it stands, with any chosen models applied."""
        return self.cfg.with_models(self.models)

    def set_model(self, engine: str, model: str) -> dict:
        if engine not in self.cfg.engines:
            return {"ok": False, "error": "no engine named %r" % engine}
        try:
            offered = [m["name"] for m in engines.available(self.cfg.engine(engine))]
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        if model not in offered:
            return {"ok": False,
                    "error": "%r does not serve %r (it has: %s)"
                             % (engine, model, ", ".join(offered) or "nothing")}
        self.models[engine] = model
        self._models_path().parent.mkdir(parents=True, exist_ok=True)
        self._models_path().write_text(json.dumps(self.models, indent=2))
        self.report("engine %s -> %s" % (engine, model))
        return {"ok": True, "engine": engine, "model": model}

    def list_models(self) -> dict:
        out = {}
        for name, eng in sorted(self.cfg.engines.items()):
            row = {"device": eng.device, "endpoint": eng.endpoint,
                   "current": self.models.get(name, eng.model)}
            try:
                row["available"] = engines.available(eng)
            except RuntimeError as exc:
                row["error"] = str(exc)
            out[name] = row
        return {"ok": True, "engines": out}

    # -- reporting ------------------------------------------------------
    def report(self, line: str) -> None:
        stamped = "%s  %s" % (time.strftime("%H:%M:%S"), line)
        with self.lock:
            self.log.append(stamped)
            del self.log[:-200]
        print(stamped, flush=True)

    # -- the queue ------------------------------------------------------
    def _worker(self) -> None:
        while True:
            session = self.jobs.get()
            with self.lock:
                self.busy = session.id
            try:
                pipeline.process(session, self.active(), self.report)
            except Exception:                            # noqa: BLE001
                self.report(traceback.format_exc())
            finally:
                with self.lock:
                    self.busy = ""

    # -- commands -------------------------------------------------------
    def source(self):
        src = self.sources.get(self.source_name)
        if src is None:
            raise RuntimeError("source %r is not installed" % self.source_name)
        return src

    # -- dictation and voice command -------------------------------------
    #
    # ⚠️ NEITHER A Stage NOR A Sink. See geshtu/plugins.py and
    # geshtu/commandes.py's own docstring: a Stage only runs inside the full
    # `cfg.stages` list, a Sink only after it. A 3-8 second clip needs exactly
    # one thing done to it -- transcribe, then act -- and running
    # diarise/chapter/summarise on it would be wasted GPU calls for output
    # nobody asked for. These two methods sit beside `pipeline.process`,
    # never inside it.
    #
    # ⚠️ THE TOGGLE IS RESOLVED HERE, SERVER-SIDE, per geshtu's own rule that
    # the daemon holds the state and clients stay thin. `voix.py`'s toggle
    # lived in a client-side pidfile because voix HAD no daemon; geshtu does,
    # and the CLI talks to it synchronously on every invocation (`cli.call()`
    # is one request, one response, never cached) -- so a daemon-side toggle
    # is never stale, unlike `tray.py`'s own 3-second status poll, which is
    # far too slow for a press-speak-press cycle of a few seconds.

    def _capture_courte(self, nom: str) -> "models.Session":
        base = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
        root = base / ("geshtu-%s" % nom)
        root.mkdir(parents=True, exist_ok=True)
        session = models.Session(id=nom, root=root)
        target = plugins.Target("default:mic", "microphone", "mic")
        session.tracks.append(self.source().start(session, target, 0))
        return session

    def _transcris_court(self, session: "models.Session") -> str:
        """A single Whisper call, no slicing -- `Transcribe`'s silence-based
        slicing exists for long recordings; a dictation clip fits in one
        request. `pipeline.process`/`Transcribe().run()` are both skipped."""
        from geshtu import audio
        wav = session.tracks[0].segments[-1] if session.tracks[0].segments else None
        if wav is None or not wav.exists():
            return ""
        # ⚠️ 350, NOT audio.py's DEFAULT OF 120. Both are the same raw
        # signed-16-bit RMS measure with the same `data`-chunk fix; only the
        # calibration differs. 350 is `voix.py`'s own figure, measured
        # against a real Whisper hallucination on true silence ("Thank you.")
        # -- exactly the failure mode a short clip is prone to, and the
        # calibration this threshold exists to guard against.
        if audio.is_silent(wav, rms_threshold=350.0):
            return ""
        return engines.transcribe(self.cfg.engine("asr"), wav)

    def dictee(self) -> dict:
        with self.lock:
            if self.dictee_session is None:
                self.dictee_session = self._capture_courte("dictee")
                self.report("dictée : j'écoute")
                return {"ok": True, "started": True}
            session, self.dictee_session = self.dictee_session, None
        self.source_for(session).stop(session)
        texte = self._transcris_court(session)
        if not texte:
            self.report("dictée : rien entendu")
            commandes.dire("Rien entendu")
            return {"ok": True, "started": False, "text": ""}
        commandes.presse_papier(texte)
        ok = subprocess.run(["wtype", texte]).returncode == 0
        if ok:
            commandes.dire("Dicté", texte[:70])
            self.report("dictée : tapé (%d mots)" % len(texte.split()))
        else:
            commandes.dire("Frappe échouée", "texte au presse-papier")
            self.report("dictée : échec de frappe (wtype), texte au presse-papier")
        return {"ok": True, "started": False, "text": texte, "typed": ok}

    def commande(self) -> dict:
        with self.lock:
            if self.commande_session is None:
                self.commande_session = self._capture_courte("commande")
                self.report("commande : j'écoute")
                return {"ok": True, "started": True}
            session, self.commande_session = self.commande_session, None
        self.source_for(session).stop(session)
        texte = self._transcris_court(session)
        if not texte:
            self.report("commande : rien entendu")
            commandes.dire("Rien entendu")
            return {"ok": True, "started": False, "text": ""}
        commandes.presse_papier(texte)

        table, chemin = commandes.charge_table()
        if table is None:
            self.report("commande : aucune table de commandes "
                        "(ni ~/.config/geshtu/commandes.json ni /etc/geshtu/commandes.json)")
            commandes.dire("Pas de table de commandes")
            return {"ok": True, "started": False, "text": texte, "matched": False}

        seuil = self.cfg.raw.get("dictation", {}).get("threshold")
        r = commandes.correspond(texte, table, chemin, self.cfg.engine("embed"), seuil)
        if not r["retenu"]:
            self.report("commande : REJET %.3f « %s » (plus proche : %s)"
                        % (r["score"], texte[:60], r["id"]))
            commandes.dire("Pas compris", texte[:70])
            return {"ok": True, "started": False, "text": texte, "matched": False}

        entree = next(c for c in table["commandes"] if c["id"] == r["id"])
        action = entree["action"]
        if "{n}" in action:
            n = commandes.nombre(texte, table)
            if n is None:
                self.report("commande : numéro manquant pour %s" % r["id"])
                commandes.dire("Numéro manquant")
                return {"ok": True, "started": False, "text": texte, "matched": False}
            action = action.replace("{n}", n)

        commandes.dire("▶ %s" % r["id"], "%s (%.2f)" % (texte[:45], r["score"]))
        ok, detail = commandes.execute(action)
        if ok:
            self.report("commande : ▶ %s" % r["id"])
        else:
            self.report("commande : « %s » a échoué : %s" % (action, detail))
            commandes.dire("Action en échec", detail or "voir le journal")
        return {"ok": True, "started": False, "text": texte, "matched": True,
                "id": r["id"], "executed": ok}

    def source_for(self, session):
        """A session records from one source; a file session has nothing to
        stop, and calling PipeWire's stop() on it would hunt for pid files
        that never existed."""
        for track in session.tracks:
            if track.kind in self.sources:
                return self.sources[track.kind]
        return self.source()

    def targets(self) -> list[dict]:
        out = []
        for name, src in self.sources.items():
            for t in src.targets():
                out.append({"source": name, "key": t.key, "label": t.label,
                            "kind": t.kind, "detail": t.detail})
        return out

    def start(self, keys: list[str], language: str) -> dict:
        with self.lock:
            if self.session is not None:
                return {"ok": False, "error": "already recording",
                        "session": self.session.id}
            root = self.cfg.sessions / time.strftime("%Y-%m-%d_%H-%M")
            root.mkdir(parents=True, exist_ok=True)
            session = models.Session(id=root.name, root=root, language=language)
            # ⚠️ A KEY MAY NAME ITS OWN SOURCE, and this is the extension
            # point rather than a special case for files.
            #
            # Some sources have nothing to enumerate because they do not tap
            # what is already there -- they MAKE it. A file is named, not
            # discovered. A meeting joiner is handed a URL, opens the call and
            # produces the audio that did not exist a second earlier. Both are
            # addressed as "<source>:<whatever that source understands>".
            #
            # ⚠️ The prefix is matched against INSTALLED SOURCE NAMES ONLY.
            # PipeWire's own targets look like "node:163", and "node" is not a
            # source, so they fall through to the default source instead of
            # being mistaken for one.
            prefix = keys[0].split(":", 1)[0] if len(keys) == 1 else ""
            if prefix and prefix in self.sources and prefix != self.source_name:
                src = self.sources[prefix]
                rest = keys[0].split(":", 1)[1]
                known = {keys[0]: plugins.Target(rest, rest, prefix)}
            else:
                src = self.source()
                known = {t.key: t for t in src.targets()}
            # ⚠️ ANYTHING THAT FAILS AFTER THE FIRST RECORDER IS LAUNCHED
            # MUST STOP IT. A half-started session leaves a live recorder, a
            # capture sink in the user's audio chooser, and a daemon that
            # believes nothing is running -- so stop() refuses to help. Seen
            # for real: a serialisation error three lines below this loop.
            try:
                for i, key in enumerate(keys):
                    target = known.get(key)
                    if target is None:
                        raise RuntimeError("unknown target %r" % key)
                    session.tracks.append(src.start(session, target, i))
                session.save()
            except Exception as exc:                     # noqa: BLE001
                try:
                    src.stop(session)
                except Exception:                        # noqa: BLE001
                    pass
                self.report("start failed, recorders stopped: %s" % exc)
                return {"ok": False, "error": str(exc)}
            self.session = session
            self.report("recording %s: %s" % (session.id, ", ".join(keys)))
            return {"ok": True, "session": session.id}

    def stop(self) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"ok": False, "error": "not recording"}
            self.source_for(session).stop(session)
            session.ended = time.time()
            self.session = None

        from geshtu import audio
        for track in session.tracks:
            if track.segments:
                track.level_db = audio.level_db(track.segments[-1])
                if track.silent:
                    # The only moment a wrong source choice can still be
                    # understood. Say it loudly and early.
                    self.report("track %s (%s) is SILENT at %.1f dB"
                                % (track.kind, track.source, track.level_db))
                else:
                    self.report("track %s (%s): %.1f dB"
                                % (track.kind, track.source, track.level_db))
        session.save()
        self.jobs.put(session)
        return {"ok": True, "session": session.id, "queued": True}

    def status(self) -> dict:
        with self.lock:
            s = self.session
            return {
                "ok": True,
                "recording": s.id if s else None,
                "duration": s.duration if s else 0.0,
                "tracks": [{"kind": t.kind, "source": t.source,
                            "level_db": t.level_db} for t in (s.tracks if s else [])],
                "processing": self.busy,
                # ⚠️ MINIMAL GUI PIECES, NOT A REDESIGN. tray.py's existing
                # poll picks these up for free; no new window, no new icon.
                "dictating": self.dictee_session is not None,
                "commanding": self.commande_session is not None,
                "log": self.log[-12:],
            }

    def sessions(self) -> list[dict]:
        if not self.cfg.sessions.is_dir():
            return []
        out = []
        for d in sorted(self.cfg.sessions.iterdir()):
            if (d / "session.json").is_file():
                raw = json.loads((d / "session.json").read_text())
                out.append({"id": raw["id"], "title": raw.get("title", ""),
                            "segments": len(raw.get("segments", [])),
                            "chapters": len(raw.get("chapters", []))})
        return out

    def reprocess(self, sid: str) -> dict:
        root = self.cfg.sessions / sid
        if not (root / "session.json").is_file():
            return {"ok": False, "error": "no such session"}
        self.jobs.put(models.Session.load(root))
        return {"ok": True, "queued": True}


class Handler(socketserver.StreamRequestHandler):
    state: State

    def handle(self) -> None:
        for raw in self.rfile:
            try:
                msg = json.loads(raw)
            except ValueError:
                self._send({"ok": False, "error": "bad json"})
                continue
            try:
                self._send(self._dispatch(msg))
            except Exception as exc:                     # noqa: BLE001
                self._send({"ok": False, "error": str(exc)})

    def _send(self, obj) -> None:
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
        self.wfile.flush()

    def _dispatch(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        st = self.state
        if cmd == "status":
            return st.status()
        if cmd == "targets":
            return {"ok": True, "targets": st.targets()}
        if cmd == "start":
            return st.start(msg.get("targets") or [], msg.get("language", "en"))
        if cmd == "stop":
            return st.stop()
        if cmd == "dictee":
            return st.dictee()
        if cmd == "commande":
            return st.commande()
        if cmd == "sessions":
            return {"ok": True, "sessions": st.sessions()}
        if cmd == "reprocess":
            return st.reprocess(msg["session"])
        if cmd == "models":
            return st.list_models()
        if cmd == "set-model":
            return st.set_model(msg["engine"], msg["model"])
        if cmd == "ping":
            return {"ok": True}
        return {"ok": False, "error": "unknown command %r" % cmd}


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    cfg = config.load()
    cfg.sessions.mkdir(parents=True, exist_ok=True)
    path = socket_path()
    # ⚠️ A stale socket after a crash makes bind() fail with EADDRINUSE and the
    # daemon looks broken when nothing is wrong. Probe it, then clear it.
    if path.exists():
        import socket as _s
        probe = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
        try:
            probe.connect(str(path))
            probe.close()
            print("geshtu: a daemon is already listening on %s" % path)
            return 1
        except OSError:
            path.unlink()
    Handler.state = State(cfg)
    with Server(str(path), Handler) as server:
        os.chmod(path, 0o600)
        print("geshtu daemon on %s" % path, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
