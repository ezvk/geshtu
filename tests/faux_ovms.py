#!/usr/bin/env python3
"""A stand-in for `ovms`, enough for tests/test_serveur.py: --configure writes a
graph.pbtxt, and the server mode re-reads its config on every request, as the
real one does with --file_system_poll_wait_seconds."""
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

a = sys.argv[1:]


def opt(name, default=None):
    return a[a.index(name) + 1] if name in a else default


if "--configure" in a:
    if "--refuse" in a:
        print("error parsing options - --refuse")
        sys.exit(1)
    d = pathlib.Path(opt("--model_path"))
    (d / "graph.pbtxt").write_text('task: "%s"\ndevice: "%s"\noptions: "%s"\n' % (
        opt("--task"), opt("--target_device", ""), " ".join(a)))
    sys.exit(0)

pathlib.Path("argv.json").write_text(json.dumps(a))
config = pathlib.Path(opt("--config_path"))


def noms():
    try:
        return [m["config"]["name"] for m in json.loads(config.read_text())["model_config_list"]]
    except (OSError, ValueError, KeyError):
        return []


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({n: {"model_version_status": [{"version": "1", "state": "AVAILABLE"}]}
                    for n in noms()})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if req.get("model") not in noms():
            self.send_response(404)
            self.end_headers()
            return
        texte = "%s: %s" % (req["model"], req["messages"][-1]["content"])
        self._json({"choices": [{"message": {"content": texte}, "finish_reason": "stop"}]})


print("[INFO] REST server on", opt("--rest_port"), flush=True)
ThreadingHTTPServer(("127.0.0.1", int(opt("--rest_port"))), H).serve_forever()
