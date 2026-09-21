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
import re
import signal
import subprocess
import sys
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
        """One target per NODE, and emphatically not one per application name.

        ⚠️ SEVERAL INSTANCES OF THE SAME PROGRAM SHARE A node.name. Measured on
        a real desktop: three Moonlight windows, all called "Moonlight", each a
        separate node with its own ports. An earlier version collapsed them
        into one entry on the theory that duplicates were noise. They were not
        -- they were three running programs, and merging them meant a request
        to record one of them would have tapped all three. That is precisely
        the leakage this source exists to prevent.

        So an application target is addressed by node id, and derivation links
        PORT IDS rather than port names: `pw-link Moonlight:output_FL ...`
        cannot say WHICH Moonlight.
        """
        nodes = _nodes()
        out: list[plugins.Target] = []
        for obj in _dump():
            props = ((obj.get("info") or {}).get("props") or {})
            cls = props.get("media.class")
            node = props.get("node.name") or ""
            if not node or cls not in ("Audio/Source", "Audio/Sink"):
                continue
            label = props.get("node.description") or node
            # ⚠️ Devices keep a stable key across reboots -- numeric ids do not
            # survive a restart, names do. Application streams are the
            # opposite case: they do not outlive their process, so there is
            # nothing to remember and precision wins.
            if cls == "Audio/Source" and "monitor" not in node:
                out.append(plugins.Target(node, label, "mic"))
            elif cls == "Audio/Sink" and node != SINK:
                out.append(plugins.Target(node, label, "output"))

        counts: dict[str, int] = {}
        for props in nodes.values():
            name = props.get("node.name") or ""
            counts[name] = counts.get(name, 0) + 1

        # ⚠️ ONLY NODES THAT ACTUALLY HAVE OUTPUT PORTS. Measured on utu:
        # four nodes named "Moonlight" existed, and only two of them carried
        # any port -- the other two cannot be recorded at all. Offering them
        # is worse than not listing them, because the user picks one, the
        # derivation silently produces nothing, and the recording is an hour
        # of the capture sink's own silence.
        with_ports = _nodes_with_ports()
        for nid, props in sorted(nodes.items()):
            if nid not in with_ports:
                continue
            node = props.get("node.name") or ""
            app = props.get("application.name") or node
            media = (props.get("media.name") or "").strip()
            detail = media[:70]
            if counts.get(node, 0) > 1:
                # ⚠️ Three identical windows are genuinely indistinguishable
                # from the outside; the node id is the only handle there is.
                # Showing it beats silently picking one of them.
                detail = ("%s #%d" % (media, nid)).strip()
            out.append(plugins.Target("node:%d" % nid, app, "app", detail))
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

    def _derive(self, node_id: int, session) -> int:
        """Add links from ONE node's output ports to the tap. Non-destructive.

        ⚠️ BY PORT ID, NOT BY PORT NAME. Port names carry the node NAME, which
        several instances of the same program share -- linking by name would
        tap every one of them. Ids address exactly one.

        A link that already exists makes pw-link fail; that is expected,
        because the watcher below reruns this every two seconds.
        """
        tap = {p["name"]: p["id"] for p in _ports_of_name(SINK, "in")}
        made = []
        for port in _ports_of_node(node_id, "out"):
            channel = port["name"].rsplit("_", 1)[-1]
            dst = tap.get("playback_" + (channel if channel in ("FL", "FR") else "FL"))
            if dst is None:
                continue
            if subprocess.run(["pw-link", str(port["id"]), str(dst)],
                              capture_output=True).returncode == 0:
                made.append("%d\t%d" % (port["id"], dst))
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

        node_id = None
        if target.kind == "app":
            node_id = int(target.key.split(":", 1)[1])
            if node_id not in _nodes():
                raise RuntimeError("that stream is gone -- list the targets again")
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
            props = _nodes().get(node_id, {})
            # ⚠️ ZERO LINKS IS A HARD FAILURE, NEVER A WARNING. Without a
            # link the recorder happily captures the capture sink's own
            # silence for as long as you let it, and every level readout
            # along the way looks healthy. Refuse now.
            if self._derive(node_id, session) == 0:
                raise RuntimeError(
                    "could not derive that stream -- nothing was linked")
            self._watch(node_id, props, session)

        return Track(kind=target.kind, source=target.label, segments=[wav])

    def _watch(self, node_id: int, props: dict, session) -> None:
        """Re-derive every two seconds.

        ⚠️ WHY A LOOP AND NOT A SINGLE LINK. A reloaded tab, a restarted call
        or an advert slotted in opens a NEW node; the original link dies with
        the old one and from then on you record silence, while everything
        still looks healthy.

        ⚠️ AND IT FOLLOWS THE NODE, NOT THE PROCESS. Following the process id
        is the obvious way to survive a reload -- and it is wrong here.
        Measured: four Firefox streams, four different tabs, ALL on pid 40428.
        Following that pid would quietly fold every other tab into the
        recording, which is the exact leakage this source exists to prevent.

        So: the node while it lives, and if it dies, one attempt to re-acquire
        by (process, stream title) -- which catches a reload without catching
        its neighbours.
        """
        spec = {
            "node": node_id,
            "pid": props.get("application.process.id"),
            "media": props.get("media.name"),
        }
        script = session.root / "watch.py"
        script.write_text(WATCHER)
        proc = subprocess.Popen(
            [sys.executable, str(script), str(session.root), json.dumps(spec)],
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


def _nodes() -> dict[int, dict]:
    """Every application stream currently playing, by node id."""
    found = {}
    for obj in _dump():
        props = ((obj.get("info") or {}).get("props") or {})
        if props.get("media.class") != "Stream/Output/Audio":
            continue
        name = props.get("node.name") or ""
        # our own tools: listing them would invite recording ourselves
        if not name or name.startswith(("pw-record", "pw-play", "pw-cat", "geshtu")):
            continue
        found[obj["id"]] = props
    return found


def _nodes_with_ports() -> set[int]:
    """Node ids that own at least one output port -- i.e. that can be tapped."""
    found = set()
    for obj in _dump():
        if not obj.get("type", "").endswith("Port"):
            continue
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("port.direction") == "out" and props.get("node.id") is not None:
            found.add(props["node.id"])
    return found


def _ports_of_node(node_id: int, direction: str) -> list[dict]:
    out = []
    for obj in _dump():
        if not obj.get("type", "").endswith("Port"):
            continue
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("node.id") == node_id and props.get("port.direction") == direction:
            out.append({"id": obj["id"], "name": props.get("port.name") or ""})
    return out


def _ports_of_name(node_name: str, direction: str) -> list[dict]:
    ids = [o["id"] for o in _dump()
           if (((o.get("info") or {}).get("props") or {}).get("node.name") == node_name)]
    out = []
    for nid in ids:
        out.extend(_ports_of_node(nid, direction))
    return out


WATCHER = """import json
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])
spec = json.loads(sys.argv[2])
SINK = "%s"


def dump():
    try:
        return json.loads(subprocess.run(["pw-dump"], capture_output=True,
                                         text=True).stdout)
    except ValueError:
        return []


def resolve(objs, props_of):
    \"\"\"The node to derive right now: the one chosen, or its replacement
    after a reload -- same process AND same title, never the process alone.\"\"\"
    live = set()
    for oid, p in props_of.items():
        if p.get("media.class") == "Stream/Output/Audio":
            live.add(oid)
    if spec["node"] in live:
        return spec["node"]
    for oid in sorted(live):
        p = props_of[oid]
        if spec.get("pid") is not None:
            if p.get("application.process.id") != spec["pid"]:
                continue
        if spec.get("media") and p.get("media.name") == spec["media"]:
            spec["node"] = oid
            return oid
    return None


while (root / "watch.pid").exists():
    objs = dump()
    props_of = {o.get("id"): ((o.get("info") or {}).get("props") or {}) for o in objs}
    node = resolve(objs, props_of)
    if node is not None:
        tap = {}
        for o in objs:
            if not o.get("type", "").endswith("Port"):
                continue
            p = (o.get("info") or {}).get("props") or {}
            holder = props_of.get(p.get("node.id"), {})
            if holder.get("node.name") == SINK and p.get("port.direction") == "in":
                tap[p.get("port.name")] = o["id"]
        for o in objs:
            if not o.get("type", "").endswith("Port"):
                continue
            p = (o.get("info") or {}).get("props") or {}
            if p.get("node.id") != node or p.get("port.direction") != "out":
                continue
            chan = (p.get("port.name") or "").rsplit("_", 1)[-1]
            dst = tap.get("playback_" + (chan if chan in ("FL", "FR") else "FL"))
            if dst is None:
                continue
            r = subprocess.run(["pw-link", str(o["id"]), str(dst)],
                               capture_output=True)
            if r.returncode == 0:
                with (root / "links.tsv").open("a") as fh:
                    fh.write("%%d\\t%%d\\n" %% (o["id"], dst))
    time.sleep(2)
""" % SINK


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text)[:32].strip("-") or "track"


PLUGIN = PipeWireSource
