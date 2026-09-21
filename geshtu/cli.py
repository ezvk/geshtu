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
  geshtu daemon                    run the daemon in the foreground

Options:
  --lang <fr|en>                   language of the summary (default: en)

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


def hms(seconds: float) -> str:
    s = int(seconds)
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


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

    if cmd == "models":
        r = call({"cmd": "models"})
        for name, row in r.get("engines", {}).items():
            print("  %-7s %-4s %s" % (name, row["device"], row["endpoint"]))
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
