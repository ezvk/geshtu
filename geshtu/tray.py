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

# ⚠️ UN SEUL CHEMIN, ET LE JOURNAL DU SHELL EST TROMPEUR SUR CE POINT.
# Il consigne « tray probe failed ... /org/ayatana/NotificationItem : Object
# does not exist » -- ce n est PAS une panne : l hote essaie d abord le chemin
# de libayatana, echoue, puis retombe sur celui de la spec et enregistre.
#
# ⚠️ ET EXPOSER LES DEUX EST PIRE : mesure du 2026-09-21, l hote sonde les
# deux, les trouve tous les deux, et enregistre DEUX items -- deux icones
# identiques dans la barre. Une ligne d erreur dans un journal n est pas une
# invitation a la faire taire.
PATH = "/StatusNotifierItem"
WATCHER = "org.kde.StatusNotifierWatcher"
# ⚠️ UN CHEMIN ABSOLU, PAS UN NOM DE THEME, ET C EST CE QUI DECIDE DE LA
# PLACE DE L ICONE.
#
# ezvk : « tu l as mis dans un tiroir, faut pas ». Comparaison directe avec
# LocalSend, qui lui est dans la barre -- memes Category et Status, rien dans
# le protocole ne distingue les deux. La seule difference est la :
#
#     LocalSend  IconName = "/nix/store/.../assets/img/logo-32-w.png"
#     geshtu     IconName = "audio-input-microphone-symbolic"
#
# C est l HOTE qui resout un nom de theme, pas nous -- d ou le fait que
# mettre adwaita-icon-theme dans notre propre fermeture n y ait jamais rien
# change. Un nom qu il ne resout pas donne une icone sans image, et il la
# range ailleurs.
#
# → On livre les fichiers et on donne leur chemin. `GESHTU_ICONS` est pose
#   par l empaquetage ; a defaut on retombe sur les noms de theme, ce qui
#   reste correct pour un `pip install` sur un bureau ordinaire.

def _icone(nom: str, repli: str) -> str:
    dossier = os.environ.get("GESHTU_ICONS")
    if dossier:
        chemin = os.path.join(dossier, nom + ".svg")
        if os.path.exists(chemin):
            return chemin
    ici = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    chemin = os.path.join(ici, nom + ".svg")
    return chemin if os.path.exists(chemin) else repli


IDLE = _icone("geshtu-idle", "audio-input-microphone-symbolic")
LIVE = _icone("geshtu-recording", "media-record-symbolic")

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
        # ⚠️ MINIMAL GUI PIECES, NOT A REDESIGN. `listening` is a superset of
        # `recording` -- true during a meeting recording OR a dictation OR a
        # voice command -- and drives the ICON (LIVE vs IDLE). `recording`
        # stays specific because only a meeting has a `duration` to show; the
        # existing LIVE icon and NeedsAttention status are reused as-is, no
        # new asset drawn.
        self.listening = False
        self.busy = False
        self.summary = "idle"
        self.status = "Active"

        self.node = Gio.DBusNodeInfo.new_for_xml(XML)
        self.conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self.conn.register_object(PATH, self.node.interfaces[0],
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
            return GLib.Variant("s", LIVE if self.listening else IDLE)
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
        self.conn.emit_signal(None, PATH, "org.kde.StatusNotifierItem",
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
        was_listening, was_status = self.listening, self.status
        self.recording = bool(r.get("recording"))
        self.busy = bool(r.get("processing"))
        # ⚠️ MINIMAL GUI PIECES, NOT A REDESIGN: no new icon asset, no new
        # window. `dictating`/`commanding` share the exact LIVE-icon /
        # NeedsAttention machinery already built for meeting recording --
        # both already mean "something is capturing audio right now".
        dictating = bool(r.get("dictating"))
        commanding = bool(r.get("commanding"))
        self.listening = self.recording or dictating or commanding
        if self.recording:
            self.summary = "recording — %s" % hms(r.get("duration", 0))
            self.status = "NeedsAttention"
        elif dictating:
            self.summary = "dictée en cours…"
            self.status = "NeedsAttention"
        elif commanding:
            self.summary = "commande…"
            self.status = "NeedsAttention"
        elif self.busy:
            self.summary = "processing %s" % r["processing"]
            self.status = "Active"
        else:
            self.summary = "idle"
            self.status = "Active"
        if self.listening != was_listening:
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
