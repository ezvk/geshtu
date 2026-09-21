"""The tray icon.

⚠️ A SEPARATE PROCESS FROM THE WINDOW, and not as a matter of taste.
libayatana-appindicator builds its menu with GTK 3; the window is GTK 4. The
two cannot be loaded into one process — gi refuses the second
require_version. So the indicator launches the window rather than containing
it, which suits an architecture where neither holds any state: both ask the
daemon.

⚠️ AND THE HOST MAY HIDE IT. Some shells put tray items in a drawer rather
than in the bar; a registered item that nobody can see looks exactly like an
item that failed to register. The check that settles it is the bus, not the
screen:

    busctl --user call org.kde.StatusNotifierWatcher /StatusNotifierWatcher \
      org.freedesktop.DBus.Properties Get ss \
      org.kde.StatusNotifierWatcher RegisteredStatusNotifierItems
"""
from __future__ import annotations

import subprocess
import threading

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")

from gi.repository import AyatanaAppIndicator3 as Indicator  # noqa: E402
from gi.repository import GLib  # noqa: E402
from gi.repository import Gtk   # noqa: E402

from geshtu.cli import call     # noqa: E402
from geshtu.cli import hms      # noqa: E402

IDLE = "audio-input-microphone-symbolic"
LIVE = "media-record-symbolic"


class Tray:

    def __init__(self):
        self.recording = False
        self.busy = False
        self.summary = "…"
        self.chosen: str | None = None
        self.targets: list[dict] = []
        self.signature = None
        self.language = "en"

        self.ind = Indicator.Indicator.new(
            "geshtu", IDLE, Indicator.IndicatorCategory.APPLICATION_STATUS)
        self.ind.set_status(Indicator.IndicatorStatus.ACTIVE)
        self.ind.set_title("geshtu")
        self.ind.set_attention_icon_full(LIVE, "recording")
        self.rebuild()
        GLib.timeout_add_seconds(3, self.tick)

    def rebuild(self) -> None:
        menu = Gtk.Menu()
        head = Gtk.MenuItem(label=self.summary)
        head.set_sensitive(False)
        menu.append(head)
        menu.append(Gtk.SeparatorMenuItem())

        if self.busy:
            item = Gtk.MenuItem(label="Working on the accelerator…")
            item.set_sensitive(False)
            menu.append(item)
        else:
            item = Gtk.MenuItem(
                label="Stop and summarise" if self.recording else "Start recording")
            item.connect("activate", self.toggle)
            item.set_sensitive(self.recording or self.chosen is not None)
            menu.append(item)

        menu.append(Gtk.SeparatorMenuItem())
        sources = Gtk.MenuItem(label="Source")
        sub = Gtk.Menu()
        # ⚠️ RADIO, NOT CHECK, and the microphone kept out of the list. One
        # records ONE source; whether one's own voice goes in with it is a
        # different question, and putting both in the same list makes two
        # unlike decisions look alike.
        first = None
        rows = list(self.targets)
        for t in rows:
            if t["kind"] == "app":
                text = "%s — %s" % (t["label"], t["detail"] or "")
            else:
                text = t["label"]
            entry = Gtk.RadioMenuItem.new_with_label([], text[:58].strip(" —"))
            if first is None:
                first = entry
            else:
                entry.join_group(first)
            entry.set_active(t["key"] == self.chosen)
            entry.set_sensitive(not self.recording)
            entry.connect("toggled", self.pick, t["key"])
            sub.append(entry)
        if not rows:
            empty = Gtk.MenuItem(label="nothing is playing")
            empty.set_sensitive(False)
            sub.append(empty)
        sources.set_submenu(sub)
        sources.set_sensitive(not self.recording)
        menu.append(sources)


        menu.append(Gtk.SeparatorMenuItem())
        window = Gtk.MenuItem(label="Open the window…")
        window.connect("activate", lambda *_: subprocess.Popen(
            ["geshtu-gui"], start_new_session=True))
        menu.append(window)
        quit_ = Gtk.MenuItem(label="Quit the indicator")
        quit_.connect("activate", lambda *_: Gtk.main_quit())
        menu.append(quit_)

        menu.show_all()
        self.ind.set_menu(menu)

    def pick(self, item, key) -> None:
        if item.get_active():
            self.chosen = key

    def toggle(self, _item) -> None:
        if self.recording:
            call({"cmd": "stop"})
        else:
            call({"cmd": "start", "targets": [self.chosen],
                  "language": self.language})
        self.tick()

    def tick(self) -> bool:
        def work():
            status = call({"cmd": "status"})
            targets = call({"cmd": "targets"}).get("targets", [])
            GLib.idle_add(self.apply, status, targets)
        threading.Thread(target=work, daemon=True).start()
        return True

    def apply(self, status, targets) -> bool:
        self.recording = bool(status.get("recording"))
        self.busy = bool(status.get("processing"))
        self.targets = targets
        # ⚠️ Keep the selection by KEY: the list moves as soon as an
        # application starts or stops, and keeping a position would silently
        # record something else.
        live = {t["key"] for t in targets}
        if self.chosen not in live:
            # ⚠️ Keep the choice by KEY and fall back visibly: the list moves
            # as soon as an application starts or stops, and silently sliding
            # onto a neighbour would record the wrong thing.
            self.chosen = sorted(live)[0] if live else None
        if self.recording:
            self.summary = "recording — %s" % hms(status.get("duration", 0))
        elif self.busy:
            self.summary = "processing %s" % status["processing"]
        else:
            self.summary = "idle"
        self.ind.set_status(Indicator.IndicatorStatus.ATTENTION if self.recording
                            else Indicator.IndicatorStatus.ACTIVE)
        self.ind.set_label(hms(status.get("duration", 0)) if self.recording else "",
                           "00:00:00")
        sig = "%s|%s|%s|%s" % (self.recording, self.busy, self.summary,
                               [t["key"] for t in targets])
        if sig != self.signature:
            self.signature = sig
            self.rebuild()
        return False


def main() -> int:
    Tray()
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
