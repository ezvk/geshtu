"""The command line client. It talks to the daemon and formats the answer."""
from __future__ import annotations

import json
import socket
import sys

from geshtu import daemon

USAGE = """geshtu — self-hosted meeting intelligence

  geshtu targets                   what can be recorded right now
  geshtu start <key> [<key>...]    start recording the given targets
  geshtu start file:<path>         run the same chain on an existing media file
  geshtu stop                      stop, then transcribe and summarise
  geshtu status                    what is recording or processing
  geshtu sessions                  past sessions
  geshtu reprocess <id>            run the pipeline again on a session
  geshtu models                    engines, their device, and what they serve
  geshtu model <engine> <name>     point an engine at another model
  geshtu models add <dir|hf:id>    serve a model dropped in the repository
  geshtu models remove <name>      stop serving it
  geshtu dictee                     press once to start, again to type
  geshtu commande                   press once to start, again to execute
  geshtu bilan                      score history for the command table
  geshtu daemon                    run the daemon in the foreground

Options:
  --lang <fr|en>                   language of the summary (default: en)

`dictee` and `commande` are toggles, meant for a keyboard shortcut: the first
invocation starts listening, the second stops, transcribes and acts. Whisper
always detects the language -- there is no override, on purpose (see
geshtu/engines.py).

An application target only exists while it is playing. Start the playback
first, then list the targets.
"""


def call(msg: dict) -> dict:
    path = daemon.socket_path()
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(path))
    except OSError:
        raise SystemExit(
            "no daemon on %s -- start it with `geshtu daemon`" % path)
    with sock:
        sock.sendall((json.dumps(msg) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data or b"{}")


def _bascule(nom: str, cmd: str) -> int:
    """Shared body of `geshtu dictee`/`geshtu commande`: guard against a
    mango key-repeat storm BEFORE the daemon is even contacted, then a single
    synchronous request that the daemon itself resolves start-vs-stop on."""
    from geshtu import keyguard
    if keyguard.repete(nom):
        return 0
    r = call({"cmd": cmd})
    if not r.get("ok"):
        print(r.get("error", "failed"))
        return 1
    if r.get("started"):
        print("%s : j'écoute -- rappuie pour terminer" % nom)
        return 0
    if not r.get("text"):
        print("rien entendu")
        return 0
    print(r["text"])
    if "typed" in r:
        print("tapé" if r["typed"] else "échec de frappe -- texte au presse-papier")
    elif "matched" in r:
        print(("▶ %s" % r["id"]) if r.get("matched") else "pas compris")
    return 0


def hms(seconds: float) -> str:
    s = int(seconds)
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


def _models_add(args: list[str]) -> int:
    """geshtu models add <dir|hf:org/model> [--name N] [--task T] [-- ...]

    Anything after a bare `--` is handed to the model server untouched, which
    is how task-specific options such as --max_prompt_len travel without this
    command having to know what they mean.
    """
    from geshtu import config as _c
    from geshtu import repository
    if not args:
        print(_models_add.__doc__)
        return 1
    source, args = args[0], args[1:]
    extra: list[str] = []
    if "--" in args:
        i = args.index("--")
        extra, args = args[i + 1:], args[:i]
    opts = dict(zip(args[::2], args[1::2]))
    name = opts.get("--name") or source.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    task = opts.get("--task", "text_generation")
    cfg = _c.load()
    if source.startswith("hf:"):
        path = repository.pull(cfg, source[3:], name, task, extra)
        print("pulled into %s" % path)
        source = path
    print(repository.add(cfg, source, name, task, extra))
    print("the server picks it up on its next poll "
          "(--file_system_poll_wait_seconds)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    language = "en"
    if "--lang" in args:
        i = args.index("--lang")
        language = args[i + 1] if len(args) > i + 1 else "en"
        del args[i:i + 2]
    if not args:
        print(USAGE)
        return 1
    cmd, rest = args[0], args[1:]

    if cmd == "daemon":
        return daemon.main()

    if cmd == "targets":
        r = call({"cmd": "targets"})
        for t in r.get("targets", []):
            print("  %-28s %-7s %s" % (t["key"], t["kind"],
                                       t["detail"] or t["label"]))
        if not r.get("targets"):
            print("  nothing to record (an app target only exists while it plays)")
        return 0

    if cmd == "start":
        if not rest:
            print("usage: geshtu start <key> [<key>...]   (see `geshtu targets`)")
            return 1
        r = call({"cmd": "start", "targets": rest, "language": language})
        print(r.get("error") or ("recording %s" % r.get("session")))
        return 0 if r.get("ok") else 1

    if cmd == "stop":
        r = call({"cmd": "stop"})
        print(r.get("error") or ("stopped %s, processing queued" % r.get("session")))
        return 0 if r.get("ok") else 1

    if cmd == "status":
        r = call({"cmd": "status"})
        if r.get("recording"):
            print("recording %s  %s" % (r["recording"], hms(r["duration"])))
            for t in r["tracks"]:
                print("  track %-7s %s" % (t["kind"], t["source"]))
        elif r.get("processing"):
            print("processing %s" % r["processing"])
        else:
            print("idle")
        for line in r.get("log", []):
            print("  %s" % line)
        return 0

    if cmd == "sessions":
        for s in call({"cmd": "sessions"}).get("sessions", []):
            print("  %-20s %-3d chapters  %s"
                  % (s["id"], s["chapters"], s["title"]))
        return 0

    if cmd == "models" and rest[:1] == ["add"]:
        return _models_add(rest[1:])

    if cmd == "models" and rest[:1] == ["remove"]:
        if len(rest) < 2:
            print("usage: geshtu models remove <name>")
            return 1
        from geshtu import config as _c
        from geshtu import repository
        print(repository.remove(_c.load(), rest[1]))
        return 0

    if cmd == "models":
        r = call({"cmd": "models"})
        # ⚠️ « declared » N EST PAS UN ORNEMENT. `device` est ce que la
        # configuration AFFIRME, pas ce que le serveur fait : rien ici ne peut
        # l interroger, et un champ laisse derriere apres un changement de
        # cote serveur ment sans que rien ne le signale. Le dire dans
        # l en-tete coute une ligne et evite de croire une etiquette.
        print("  engine  declared  endpoint")
        for name, row in r.get("engines", {}).items():
            print("  %-7s %-9s %s" % (name, row["device"], row["endpoint"]))
            if row.get("error"):
                print("      unreachable: %s" % row["error"])
                continue
            for m in row.get("available", []):
                mark = " <-" if m["name"] == row["current"] else ""
                note = "" if m["state"] == "AVAILABLE" else "  (%s)" % m["state"]
                print("      %-32s%s%s" % (m["name"], note, mark))
        return 0

    if cmd == "model":
        if len(rest) < 2:
            print("usage: geshtu model <engine> <name>   (see `geshtu models`)")
            return 1
        r = call({"cmd": "set-model", "engine": rest[0], "model": rest[1]})
        print(r.get("error") or ("%s -> %s" % (r["engine"], r["model"])))
        return 0 if r.get("ok") else 1

    if cmd == "dictee":
        return _bascule("dictee", "dictee")

    if cmd == "commande":
        return _bascule("commande", "commande")

    if cmd == "bilan":
        from geshtu import commandes
        commandes.bilan()
        return 0

    if cmd == "reprocess":
        if not rest:
            print("usage: geshtu reprocess <id>")
            return 1
        r = call({"cmd": "reprocess", "session": rest[0]})
        print(r.get("error") or "queued")
        return 0 if r.get("ok") else 1

    print(USAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
