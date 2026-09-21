"""Capture from PipeWire: a microphone, the system output, or ONE application.

⚠️ THE APPLICATION CASE IS THE WHOLE POINT, AND IT IS NOT OBVIOUS.

Recording the system output mixes everything that plays; a microphone picks up
the whole room. To capture one browser tab and nothing else you have to reach
the stream itself -- and there are two dead ends on the way:

  1. You CANNOT point a recorder at a stream. A capture linked only to another
     stream has no clock driver, so nothing flows: the measured result is a
     44-byte file with n_samples 0.
  2. You CANNOT move the application to a capture sink either -- it would stop
     being audible.

So: create a silent null sink, and DERIVE the stream into it by ADDING a link,
leaving the existing link to the speakers untouched.

    Firefox:output_FL --+--> Speaker:playback_FL      (untouched, still heard)
                        +--> geshtu-tap:playback_FL   (added, recorded)

⚠️ MEASURED, WITH A CONTROL -- same sink, same stream playing, only the link
differs:   link present -24.1 dB     link cut -91.0 dB (silence)

And with a deliberate second stream playing to the same default sink: the
derived 440 Hz reads -24.1 dB while the undervied 1200 Hz reads -55.3 dB --
which is exactly the bandpass filter's own leakage on a pure 440 Hz tone.
Nothing of the other stream gets in.

The control is not a formality. Three earlier arrangements produced audio
while not working at all, because a recorder whose target does not exist falls
back to the default sink IN SILENCE. Without the negative case you cannot tell
a tap from a fallback.

⚠️ THE SINK EXISTS ONLY WHILE RECORDING. Declared permanently, it shows up in
every audio chooser, where one click sends the whole machine's sound into a
void. No property hides a node from those UIs: node.disabled prevents creation,
priority.session only demotes, and neither node.hidden nor node.virtual exists.
So it is created at start and destroyed at stop.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import subprocess
import time

from geshtu import plugins
from geshtu.models import Track

SINK = "geshtu-tap"

SINK_ARGS = (
    "{ factory.name=support.null-audio-sink"
    " node.name=%s"
    ' node.description="geshtu (recording in progress)"'
    " media.class=Audio/Sink"
    " object.linger=true"
    " audio.position=[FL,FR]"
    " priority.session=0"
    " priority.driver=0 }" % SINK
)


def _run(*cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def _dump() -> list:
    try:
        return json.loads(_run("pw-dump"))
    except ValueError:
        return []


class PipeWireSource:
    name = "pipewire"

    # -- discovery ------------------------------------------------------
    def targets(self) -> list[plugins.Target]:
        out: list[plugins.Target] = []
        for obj in _dump():
            props = ((obj.get("info") or {}).get("props") or {})
            cls = props.get("media.class")
            node = props.get("node.name") or ""
            if not node:
                continue
            label = props.get("node.description") or node
            if cls == "Audio/Source" and "monitor" not in node:
                out.append(plugins.Target(node, label, "mic"))
            elif cls == "Audio/Sink" and node != SINK:
                out.append(plugins.Target(node, label, "output"))
            elif cls == "Stream/Output/Audio":
                if node.startswith(("pw-record", "pw-play", "pw-cat", "geshtu")):
                    continue
                app = props.get("application.name") or node
                media = (props.get("media.name") or "").strip()
                out.append(plugins.Target(node, app, "app", media[:70]))
        return out

    # -- the tap --------------------------------------------------------
    def _sink_present(self) -> bool:
        return any(x.strip().startswith(SINK + ":")
                   for x in _run("pw-link", "-i").splitlines())

    def _create_sink(self) -> bool:
        """⚠️ WAIT FOR THE PORTS, do not trust pw-cli's exit code. Creation is
        asynchronous; a recorder started too early does not find its target and
        falls back to the default sink in silence."""
        if self._sink_present():
            return True
        subprocess.run(["pw-cli", "create-node", "adapter", SINK_ARGS],
                       capture_output=True)
        for _ in range(30):
            if self._sink_present():
                return True
            time.sleep(0.1)
        return False

    def _destroy_sink(self) -> None:
        for obj in _dump():
            props = ((obj.get("info") or {}).get("props") or {})
            if props.get("node.name") == SINK:
                subprocess.run(["pw-cli", "destroy", str(obj["id"])],
                               capture_output=True)

    def _derive(self, node: str, session) -> int:
        """Add a link from the application's ports to the tap. Non-destructive.

        A link that already exists makes pw-link fail with "File exists"; that
        is expected, because the watcher below reruns this every two seconds.
        """
        made = []
        for port in _run("pw-link", "-o").splitlines():
            port = port.strip()
            if not port.startswith(node + ":"):
                continue
            channel = port.rsplit("_", 1)[-1]
            dst = "%s:playback_%s" % (SINK, channel if channel in ("FL", "FR") else "FL")
            if subprocess.run(["pw-link", port, dst],
                              capture_output=True).returncode == 0:
                made.append("%s\t%s" % (port, dst))
        if made:
            with (session.root / "links.tsv").open("a") as fh:
                fh.write("\n".join(made) + "\n")
        return len(made)

    def _cut_links(self, session) -> None:
        """Remove only the links WE made -- hence the journal. Cutting by
        pattern would also drop links the user or the session manager made."""
        journal = session.root / "links.tsv"
        try:
            lines = journal.read_text().splitlines()
        except OSError:
            return
        for line in dict.fromkeys(lines):
            parts = line.split("\t")
            if len(parts) == 2:
                subprocess.run(["pw-link", "-d"] + parts, capture_output=True)
        journal.unlink(missing_ok=True)

    # -- recording ------------------------------------------------------
    def start(self, session, target: plugins.Target, index: int) -> Track:
        rate, channels = 16000, 1
        wav = session.root / ("%s-%s-%03d.wav" % (target.kind, _safe(target.key), index))
        cmd = ["pw-record", "--rate", str(rate), "--channels", str(channels)]

        if target.kind == "app":
            if not self._create_sink():
                raise RuntimeError("could not create the capture sink %s" % SINK)
            cmd += ["--target", SINK, "-P", "stream.capture.sink=true"]
        elif target.kind == "output":
            # ⚠️ stream.capture.sink is tied to the MODE, never to a setting:
            # on a microphone it diverts the capture to the default output.
            cmd += ["--target", target.key, "-P", "stream.capture.sink=true"]
        else:
            cmd += ["--target", target.key]

        cmd.append(str(wav))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        (session.root / ("%s.pid" % _safe(target.key))).write_text(str(proc.pid))

        if target.kind == "app":
            self._derive(target.key, session)
            self._watch(target.key, session)

        return Track(kind=target.kind, source=target.label, segments=[wav])

    def _watch(self, node: str, session) -> None:
        """Re-derive every two seconds.

        ⚠️ WHY A LOOP AND NOT A SINGLE LINK. An application opens a NEW node
        for every stream: a reloaded tab, a restarted call, an advert slotted
        in. The original link dies with the old node, and from then on you
        record silence -- while everything still looks healthy.
        """
        script = (
            "import subprocess,sys,time,pathlib\n"
            "node,root=sys.argv[1],pathlib.Path(sys.argv[2])\n"
            "while (root/'watch.pid').exists():\n"
            "    out=subprocess.run(['pw-link','-o'],capture_output=True,text=True).stdout\n"
            "    for p in out.splitlines():\n"
            "        p=p.strip()\n"
            "        if not p.startswith(node+':'): continue\n"
            "        c=p.rsplit('_',1)[-1]\n"
            "        d='%s:playback_'+(c if c in ('FL','FR') else 'FL')\n"
            "        r=subprocess.run(['pw-link',p,d],capture_output=True)\n"
            "        if r.returncode==0:\n"
            "            (root/'links.tsv').open('a').write(p+chr(9)+d+chr(10))\n"
            "    time.sleep(2)\n" % SINK
        )
        path = session.root / "watch.py"
        path.write_text(script)
        proc = subprocess.Popen([os.sys.executable, str(path), node, str(session.root)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        (session.root / "watch.pid").write_text(str(proc.pid))

    def stop(self, session) -> None:
        pids = []
        for f in sorted(session.root.glob("*.pid")):
            try:
                pids.append(int(f.read_text().strip()))
            except (OSError, ValueError):
                pass
            f.unlink(missing_ok=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass
        # ⚠️ WAIT for the writers to exit. A WAV read before its RIFF header is
        # finalised makes the ASR return an intermittent HTTP 400.
        deadline = time.time() + 5
        while time.time() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.1)
        for pid in pids:
            if _alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        self._cut_links(session)
        self._destroy_sink()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text)[:32].strip("-") or "track"


PLUGIN = PipeWireSource
