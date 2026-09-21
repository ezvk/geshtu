"""The window. It drives the daemon and shows what the daemon says.

⚠️ IT HOLDS NO STATE OF ITS OWN. Everything comes from `status` every three
seconds, so a recording started from the tray, the keyboard or a terminal
shows up here, and one stopped elsewhere stops showing. The previous
generation of this tool recomputed status in each client, and when that
computation broke every interface went blank at once while the recording was
perfectly fine — a failure in the display is the worst place to have one,
because it makes healthy work look dead.
"""
from __future__ import annotations

import threading

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import GLib  # noqa: E402
from gi.repository import Gtk   # noqa: E402

from geshtu.cli import call     # noqa: E402
from geshtu.cli import hms      # noqa: E402

GROUPS = [("app", "Applications"), ("mic", "Microphones"), ("output", "Outputs")]


class Window(Gtk.ApplicationWindow):

    def __init__(self, app):
        super().__init__(application=app, title="geshtu")
        self.set_default_size(620, 620)
        self.chosen: set[str] = set()
        self.busy = False
        self.recording = False

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for side in ("top", "bottom", "start", "end"):
            getattr(box, "set_margin_" + side)(14)
        self.set_child(box)

        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.state = Gtk.Label(label="…", xalign=0, hexpand=True)
        head.append(self.state)
        refresh = Gtk.Button(icon_name="view-refresh-symbolic")
        refresh.set_tooltip_text("Re-read what is playing")
        refresh.connect("clicked", lambda *_: self.reload())
        head.append(refresh)
        box.append(head)

        # ---- sources -------------------------------------------------
        #
        # ⚠️ A LIST, NOT A DROPDOWN. A session records SEVERAL tracks at once
        # — the meeting in the browser and your own microphone is the normal
        # case — and they are mixed afterwards. A single-choice control would
        # quietly make that impossible.
        self.list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        scroll = Gtk.ScrolledWindow(vexpand=True, child=self.list)
        scroll.add_css_class("frame")
        box.append(scroll)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.append(Gtk.Label(label="Summary in", xalign=0))
        self.lang = Gtk.DropDown(model=Gtk.StringList.new(["english", "français"]))
        row.append(self.lang)
        box.append(row)

        self.button = Gtk.Button(label="Start recording")
        self.button.add_css_class("suggested-action")
        self.button.set_size_request(-1, 46)
        self.button.connect("clicked", self.toggle)
        box.append(self.button)

        # ---- engines -------------------------------------------------
        #
        # Folded away because it is not part of recording, but present
        # because which model is loaded changes under you.
        self.engines = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        exp = Gtk.Expander(label="Engines", child=self.engines)
        box.append(exp)
        exp.connect("notify::expanded", self.on_engines)

        self.view = Gtk.TextView(editable=False, monospace=True,
                                 wrap_mode=Gtk.WrapMode.WORD_CHAR)
        low = Gtk.ScrolledWindow(child=self.view)
        low.set_size_request(-1, 150)
        low.add_css_class("frame")
        box.append(low)

        self.reload()
        GLib.timeout_add_seconds(3, self.tick)

    # ------------------------------------------------------------- helpers
    def log(self, text: str) -> None:
        buf = self.view.get_buffer()
        buf.insert(buf.get_end_iter(), text.rstrip() + "\n")
        self.view.scroll_to_mark(
            buf.create_mark(None, buf.get_end_iter(), False), 0, False, 0, 0)

    def reload(self) -> None:
        r = call({"cmd": "targets"})
        child = self.list.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self.list.remove(child)
            child = nxt
        targets = r.get("targets", [])
        for kind, title in GROUPS:
            rows = [t for t in targets if t["kind"] == kind]
            if not rows:
                continue
            head = Gtk.Label(label=title, xalign=0)
            head.add_css_class("dim-label")
            head.set_margin_top(6)
            self.list.append(head)
            for t in rows:
                label = t["detail"] or t["label"]
                if kind != "app":
                    label = t["label"]
                check = Gtk.CheckButton(label=label[:80])
                check.set_active(t["key"] in self.chosen)
                check.connect("toggled", self.pick, t["key"])
                self.list.append(check)
        if not targets:
            self.list.append(Gtk.Label(
                label="nothing to record — an app appears only while it plays",
                xalign=0))

    def pick(self, check, key) -> None:
        if check.get_active():
            self.chosen.add(key)
        else:
            self.chosen.discard(key)

    # ------------------------------------------------------------- engines
    def on_engines(self, expander, _param) -> None:
        if not expander.get_expanded():
            return
        child = self.engines.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self.engines.remove(child)
            child = nxt
        r = call({"cmd": "models"})
        for name, row in r.get("engines", {}).items():
            line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            line.append(Gtk.Label(label="%s (%s)" % (name, row["device"]), xalign=0))
            if row.get("error"):
                warn = Gtk.Label(label="unreachable", xalign=0)
                warn.add_css_class("error")
                line.append(warn)
                self.engines.append(line)
                continue
            names = [m["name"] for m in row.get("available", [])]
            drop = Gtk.DropDown(model=Gtk.StringList.new(names or ["—"]),
                                hexpand=True)
            if row["current"] in names:
                drop.set_selected(names.index(row["current"]))
            drop.connect("notify::selected", self.set_model, name, names)
            line.append(drop)
            self.engines.append(line)

    def set_model(self, drop, _param, engine, names) -> None:
        i = drop.get_selected()
        if not 0 <= i < len(names):
            return
        r = call({"cmd": "set-model", "engine": engine, "model": names[i]})
        self.log(r.get("error") or ("%s -> %s" % (engine, names[i])))

    # ------------------------------------------------------------- actions
    def toggle(self, _button) -> None:
        if self.busy:
            return
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if not self.chosen:
            self.log("pick at least one source")
            return
        lang = "en" if self.lang.get_selected() == 0 else "fr"
        r = call({"cmd": "start", "targets": sorted(self.chosen),
                  "language": lang})
        self.log(r.get("error") or ("recording %s" % r.get("session")))
        self.tick()

    def stop(self) -> None:
        """⚠️ The daemon queues the processing and answers at once, so there is
        nothing to wait for here. That is the whole reason it exists: the
        window can be closed while an hour of audio is being transcribed."""
        r = call({"cmd": "stop"})
        self.log(r.get("error") or ("stopped %s" % r.get("session")))
        self.tick()

    # ---------------------------------------------------------------- state
    def tick(self) -> bool:
        def work():
            GLib.idle_add(self.apply, call({"cmd": "status"}))
        threading.Thread(target=work, daemon=True).start()
        return True

    def apply(self, r) -> bool:
        self.recording = bool(r.get("recording"))
        self.busy = bool(r.get("processing"))
        if self.recording:
            self.state.set_text("recording %s   %s"
                                % (r["recording"], hms(r.get("duration", 0))))
            self.button.set_label("Stop and summarise")
            self.button.remove_css_class("suggested-action")
            self.button.add_css_class("destructive-action")
        elif self.busy:
            self.state.set_text("processing %s on the accelerator" % r["processing"])
            self.button.set_label("working…")
        else:
            self.state.set_text("idle")
            self.button.set_label("Start recording")
            self.button.remove_css_class("destructive-action")
            self.button.add_css_class("suggested-action")
        self.button.set_sensitive(not self.busy)
        buf = self.view.get_buffer()
        known = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        for line in r.get("log", []):
            if line not in known:
                self.log(line)
        return False


def main() -> int:
    app = Gtk.Application(application_id="org.geshtu.Window")
    app.connect("activate", lambda a: Window(a).present())
    return app.run(None)


if __name__ == "__main__":
    raise SystemExit(main())
