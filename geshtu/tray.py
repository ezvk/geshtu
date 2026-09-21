"""The tray icon, as a StatusNotifierItem spoken directly over D-Bus.

⚠️ WHY NOT libayatana-appindicator, WHICH IS THE OBVIOUS CHOICE. Because it
does not implement `Activate`. Introspected on the real object:

    org.kde.StatusNotifierItem
      .Scroll .SecondaryActivate .XAyatanaSecondaryActivate
      (no Activate, no ItemIsMenu)

Its model is "the menu IS the interaction". A host that follows the spec calls
`Activate` on left click, gets `UnknownMethod`, and does nothing — which is
exactly what the shell log said:

    [tray] activate failed err=[org.freedesktop.DBus.Error.UnknownMethod]
           La méthode « Activate » n'existe pas

ezvk: "regarde un tray icone standard comme celui de syncthing ou localsend,
ils sont ok". They are, because they implement Activate. So does this now.

⚠️ AND IT DROPS TWO DEPENDENCIES. No GTK 3, no libayatana: the interaction is
click to open the window, middle click to start or stop. Nothing here needs a
menu, so nothing here needs a toolkit — GLib alone speaks D-Bus. That also
ends the GTK 3 / GTK 4 split that forced this into a separate process for
technical rather than useful reasons; it stays separate because a tray icon
must outlive the window, which is a better reason.
"""
from __future__ import annotations

import os
import subprocess
import threading

import gi

from gi.repository import Gio    # noqa: E402
from gi.repository import GLib   # noqa: E402

from geshtu.cli import call      # noqa: E402
from geshtu.cli import hms       # noqa: E402

assert gi  # imported for the typelib side effect

# ⚠️ DEUX CHEMINS, ET CE N EST PAS DE LA SUPERSTITION. La spec dit
# /StatusNotifierItem ; libayatana publie sous /org/ayatana/NotificationItem,
# et les hotes qui ont grandi avec lui sondent CE chemin d abord. Mesure sur
# la machine, journal du shell :
#
#     [tray] tray probe failed bus=... path=/org/ayatana/NotificationItem
#            Object does not exist at path
#
# L item etait bien enregistre et le shell ne le rendait pas comme ses
# voisins -- LocalSend, juste a cote, publie sous le chemin ayatana. On
# expose le meme objet aux deux endroits : cela ne coute rien et supprime la
# question de savoir lequel l hote prefere.
PATH = "/StatusNotifierItem"
PATH_AYATANA = "/org/ayatana/NotificationItem"
WATCHER = "org.kde.StatusNotifierWatcher"
IDLE = "audio-input-microphone-symbolic"
LIVE = "media-record-symbolic"

XML = """
<node>
  <interface name="org.kde.StatusNotifierItem">
    <property name="Category" type="s" access="read"/>
    <property name="Id" type="s" access="read"/>
    <property name="Title" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconName" type="s" access="read"/>
    <property name="AttentionIconName" type="s" access="read"/>
    <property name="OverlayIconName" type="s" access="read"/>
    <property name="IconThemePath" type="s" access="read"/>
    <property name="ItemIsMenu" type="b" access="read"/>
    <property name="Menu" type="o" access="read"/>
    <property name="ToolTip" type="(sa(iiay)ss)" access="read"/>
    <method name="Activate">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="SecondaryActivate">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="ContextMenu">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="Scroll">
      <arg name="delta" type="i" direction="in"/>
      <arg name="orientation" type="s" direction="in"/>
    </method>
    <signal name="NewIcon"/>
    <signal name="NewToolTip"/>
    <signal name="NewStatus"><arg name="status" type="s"/></signal>
  </interface>
</node>
"""


class Item:

    def __init__(self):
        self.recording = False
        self.busy = False
        self.summary = "idle"
        self.status = "Active"

        self.node = Gio.DBusNodeInfo.new_for_xml(XML)
        self.conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        for chemin in (PATH, PATH_AYATANA):
            self.conn.register_object(chemin, self.node.interfaces[0],
                                      self.on_call, self.on_get, None)
        # ⚠️ THE NAME SHAPE IS PART OF THE SPEC: hosts that predate the
        # "pass your unique name" convention look for exactly this.
        self.name = "org.kde.StatusNotifierItem-%d-1" % os.getpid()
        Gio.bus_own_name_on_connection(
            self.conn, self.name, Gio.BusNameOwnerFlags.NONE,
            lambda *_: self.register(), None)

    def register(self) -> None:
        try:
            self.conn.call_sync(
                WATCHER, "/StatusNotifierWatcher", WATCHER,
                "RegisterStatusNotifierItem", GLib.Variant("(s)", (self.name,)),
                None, Gio.DBusCallFlags.NONE, 3000, None)
        except GLib.Error as exc:
            # ⚠️ NOT FATAL, and worth saying rather than dying: a shell may
            # start after us. The watcher name appearing later is handled by
            # the retry below, and a host that never appears leaves a working
            # daemon with an invisible icon rather than no daemon at all.
            print("geshtu-tray: no tray host yet (%s)" % exc.message)

    # ------------------------------------------------------------ D-Bus
    def on_get(self, _conn, _sender, _path, _iface, prop):
        if prop == "Category":
            return GLib.Variant("s", "ApplicationStatus")
        if prop == "Id":
            return GLib.Variant("s", "geshtu")
        if prop == "Title":
            return GLib.Variant("s", "geshtu")
        if prop == "Status":
            return GLib.Variant("s", self.status)
        if prop == "IconName":
            return GLib.Variant("s", LIVE if self.recording else IDLE)
        if prop == "AttentionIconName":
            return GLib.Variant("s", LIVE)
        if prop in ("OverlayIconName", "IconThemePath"):
            return GLib.Variant("s", "")
        if prop == "ItemIsMenu":
            # ⚠️ false, AND IT MATTERS: it tells the host "I answer Activate,
            # do not go looking for a menu". An item that says true and has no
            # menu object is a dead icon.
            return GLib.Variant("b", False)
        if prop == "Menu":
            return GLib.Variant("o", "/NO_DBUSMENU")
        if prop == "ToolTip":
            return GLib.Variant("(sa(iiay)ss)",
                                (IDLE, [], "geshtu", self.summary))
        return None

    def on_call(self, _conn, _sender, _path, _iface, method, _params, invocation):
        if method == "Activate":
            subprocess.Popen(["geshtu-gui"], start_new_session=True)
        elif method == "SecondaryActivate":
            self.toggle()
        elif method == "ContextMenu":
            subprocess.Popen(["geshtu-gui"], start_new_session=True)
        invocation.return_value(None)

    def emit(self, signal, args=None) -> None:
        for chemin in (PATH, PATH_AYATANA):
            self.conn.emit_signal(None, chemin, "org.kde.StatusNotifierItem",
                                  signal, args)

    # ----------------------------------------------------------- actions
    def toggle(self) -> None:
        if self.busy:
            return
        if self.recording:
            call({"cmd": "stop"})
        else:
            # ⚠️ NO SOURCE PICKER HERE, and that is deliberate: an application
            # stream cannot be chosen from an icon, and guessing one would
            # record the wrong thing silently. Without a prior choice the
            # daemon refuses and says so; the window is where one chooses.
            call({"cmd": "start", "targets": ["default:output"],
                  "language": "fr"})
        self.tick()

    # ------------------------------------------------------------- state
    def tick(self) -> bool:
        def work():
            GLib.idle_add(self.apply, call({"cmd": "status"}))
        threading.Thread(target=work, daemon=True).start()
        return True

    def apply(self, r) -> bool:
        was_rec, was_status = self.recording, self.status
        self.recording = bool(r.get("recording"))
        self.busy = bool(r.get("processing"))
        if self.recording:
            self.summary = "recording — %s" % hms(r.get("duration", 0))
            self.status = "NeedsAttention"
        elif self.busy:
            self.summary = "processing %s" % r["processing"]
            self.status = "Active"
        else:
            self.summary = "idle"
            self.status = "Active"
        if self.recording != was_rec:
            self.emit("NewIcon")
        if self.status != was_status:
            self.emit("NewStatus", GLib.Variant("(s)", (self.status,)))
        self.emit("NewToolTip")
        return False


def main() -> int:
    item = Item()
    item.tick()
    GLib.timeout_add_seconds(3, item.tick)
    # ⚠️ Re-register when a shell restarts: noctalia, or any host, drops every
    # item when it dies, and an icon that never comes back looks like a crash
    # in this process instead of a restart in that one.
    Gio.bus_watch_name_on_connection(
        item.conn, WATCHER, Gio.BusNameWatcherFlags.NONE,
        lambda *_: item.register(), None)
    GLib.MainLoop().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
