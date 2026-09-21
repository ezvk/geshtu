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

# Streams first, because that is what one actually picks; the two fallbacks
# after, because they are what one falls back to.

class Window(Gtk.ApplicationWindow):

    def __init__(self, app):
        super().__init__(application=app, title="geshtu")
        self.set_default_size(620, 620)
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
        # ⚠️ UN SEUL CHOIX, ET LE MICRO A PART. La premiere version cochait
        # librement plusieurs lignes ; ezvk : « source selection should not be
        # a checkmark but a list with a unique selectable choice ». Il a
        # raison : on enregistre UNE source, et la question « est-ce que ma
        # voix y va aussi » n en est pas une deuxieme du meme genre. Les
        # melanger fait une liste ou deux decisions differentes se ressemblent.
        self.rows: list[dict] = []
        self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.list.add_css_class("rich-list")
        scroll = Gtk.ScrolledWindow(vexpand=True, child=self.list)
        scroll.add_css_class("frame")
        box.append(scroll)

        self.with_mic = Gtk.CheckButton(label="Also record my microphone")
        self.with_mic.set_active(True)
        box.append(self.with_mic)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.append(Gtk.Label(label="Summary in", xalign=0))
        self.lang = Gtk.DropDown(model=Gtk.StringList.new(["english", "français"]))
        row.append(self.lang)
        box.append(row)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.button = Gtk.Button(label="Start recording", hexpand=True)
        self.button.add_css_class("suggested-action")
        self.button.set_size_request(-1, 46)
        self.button.connect("clicked", self.toggle)
        actions.append(self.button)
        self.openfile = Gtk.Button(label="Open a recording…")
        self.openfile.set_tooltip_text(
            "Transcribe and summarise a file that already exists")
        self.openfile.connect("clicked", self.open_file)
        actions.append(self.openfile)
        box.append(actions)

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
        keep = self.selected_key()
        r = call({"cmd": "targets"})
        child = self.list.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self.list.remove(child)
            child = nxt
        self.rows = []
        # ⚠️ Le micro n est PAS dans la liste : il est la case au-dessous.
        targets = [t for t in r.get("targets", []) if t["kind"] != "mic"]
        for t in targets:
            if t["kind"] == "app":
                title = t["detail"] or t["label"]
                text = "%s — %s" % (t["label"], title)
            else:
                text = "%s — %s" % (t["label"], t["detail"])
            label = Gtk.Label(label=text[:90], xalign=0)
            label.set_margin_top(6)
            label.set_margin_bottom(6)
            label.set_margin_start(8)
            self.list.append(label)
            self.rows.append(t)
        if not targets:
            self.list.append(Gtk.Label(
                label="nothing is playing — a stream exists only while it plays",
                xalign=0))
            return
        for i, t in enumerate(self.rows):
            if t["key"] == keep:
                self.list.select_row(self.list.get_row_at_index(i))
                return
        self.list.select_row(self.list.get_row_at_index(0))

    def selected_key(self) -> str | None:
        row = self.list.get_selected_row()
        if row is None:
            return None
        i = row.get_index()
        return self.rows[i]["key"] if 0 <= i < len(self.rows) else None

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
            # "declared" plutot que le seul nom : le champ peut mentir.
            tag = Gtk.Label(label="%s (%s declared)" % (name, row["device"]),
                            xalign=0)
            line.append(tag)
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
        key = self.selected_key()
        if key is None:
            self.log("pick a source")
            return
        targets = [key]
        if self.with_mic.get_active():
            targets.append("default:mic")
        r = call({"cmd": "start", "targets": targets, "language": self.lang_code()})
        self.log(r.get("error") or ("recording %s" % r.get("session")))
        self.tick()

    def lang_code(self) -> str:
        return "en" if self.lang.get_selected() == 0 else "fr"

    def open_file(self, _button) -> None:
        """Run the same chain on something already recorded.

        ⚠️ IT IS A START FOLLOWED AT ONCE BY A STOP, not a separate path. A
        file has nothing to capture, so the only honest way to keep one code
        path is to let the pipeline see it as a session that is already over.
        A second path would double the number of places a recording can end
        up empty without anyone noticing.
        """
        dialog = Gtk.FileDialog(title="Open a recording")

        def chosen(dlg, res):
            try:
                gfile = dlg.open_finish(res)
            except Exception:                            # noqa: BLE001
                return                                   # cancelled
            path = gfile.get_path()
            if not path:
                return
            r = call({"cmd": "start", "targets": ["file:" + path],
                      "language": self.lang_code()})
            if not r.get("ok"):
                self.log(r.get("error", "could not open it"))
                return
            self.log("reading %s" % path)
            self.log(call({"cmd": "stop"}).get("error") or "queued for the accelerator")
            self.tick()

        dialog.open(self, None, chosen)

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
        self.openfile.set_sensitive(not self.busy and not self.recording)
        self.list.set_sensitive(not self.recording)
        self.with_mic.set_sensitive(not self.recording)
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
