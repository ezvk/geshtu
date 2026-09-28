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

from gi.repository import Gio   # noqa: E402
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
        self.traduction_active = False

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
        # ⚠️ AN EXPLICIT WAY OUT. mango draws no title bar, so the window had no
        # close button, and geshtu is a single-instance Gtk.Application: what
        # looked like closing it left it running, and "reopening" only raised
        # the same instance with its old state. ezvk: « on peut pas quit ».
        # Button + Ctrl+Q / Ctrl+W (see main()); the daemon is not affected.
        quitter = Gtk.Button(icon_name="application-exit-symbolic")
        quitter.set_tooltip_text("Quit the window (Ctrl+Q) -- the daemon keeps running")
        quitter.connect("clicked", lambda *_: app.quit())
        head.append(quitter)
        box.append(head)

        # ---- sources -------------------------------------------------
        #
        # ⚠️ UN SEUL CHOIX, ET LE MICRO A PART. La premiere version cochait
        # librement plusieurs lignes ; ezvk : « source selection should not be
        # a checkmark but a list with a unique selectable choice ». Il a
        # raison : on enregistre UNE source, et la question « est-ce que ma
        # voix y va aussi » n en est pas une deuxieme du meme genre. Les
        # melanger fait une liste ou deux decisions differentes se ressemblent.
        #
        # ⚠️ ET UN MENU DÉROULANT, PLUS UNE LISTE. ezvk, 2026-09-27 : « le
        # sélecteur d'audio ça doit être un menu déroulant pas une liste
        # d'entrées », « ça fait de la place pour la fenêtre de transcript ».
        # La place gagnée va aux sous-titres de la traduction en direct.
        self.rows: list[dict] = []
        src = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        src.append(Gtk.Label(label="Source", xalign=0))
        self.sources_menu = Gtk.DropDown(model=Gtk.StringList.new(["…"]), hexpand=True)
        src.append(self.sources_menu)
        box.append(src)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        # The SPOKEN language, separate from the summary's -- Whisper on OVMS
        # does not detect it per request (engines.transcribe()).
        row.append(Gtk.Label(label="Audio en", xalign=0))
        # ezvk, 2026-09-28 : « comme langue de source audio rajoute russe
        # chinois japonais ». Whisper les transcrit ; chacun dans sa langue.
        self.entrees = ["fr", "en", "ru", "zh", "ja"]
        self.entree = Gtk.DropDown(model=Gtk.StringList.new(
            ["français", "english", "русский", "中文", "日本語"]))
        e = call({"cmd": "get-entree"}).get("langue_entree", "fr")
        if e in self.entrees:
            self.entree.set_selected(self.entrees.index(e))
        self.entree.connect("notify::selected", self.set_entree)
        row.append(self.entree)
        row.append(Gtk.Label(label="Summary in", xalign=0))
        self.lang = Gtk.DropDown(model=Gtk.StringList.new(["english", "français"]))
        # ⚠️ READ FROM THE DAEMON, WRITTEN BACK ON CHANGE: the drop-down used to
        # come up on "english" at every launch, whatever was chosen last time.
        courante = call({"cmd": "get-language"}).get("language", "fr")
        self.lang.set_selected(0 if courante == "en" else 1)
        self.lang.connect("notify::selected", self.set_language)
        row.append(self.lang)
        row.append(Gtk.Label(label="Traduire vers", xalign=0))
        self.cibles = ["fr", "en", "es", "it", "pt"]
        self.cible = Gtk.DropDown(model=Gtk.StringList.new(
            ["français", "english", "español", "italiano", "português"]))
        c = call({"cmd": "get-cible"}).get("langue_cible", "fr")
        if c in self.cibles:
            self.cible.set_selected(self.cibles.index(c))
        self.cible.connect("notify::selected", self.set_cible)
        row.append(self.cible)
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

        # ---- read the latest summary aloud (geshtu/lecture.py) ----------
        lire = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.lire = Gtk.Button(label="▶ Lire le dernier résumé", hexpand=True)
        self.lire.set_tooltip_text("Kokoro, sur le NPU -- lecture / pause / reprise")
        self.lire.connect("clicked", self.lire_bascule)
        lire.append(self.lire)
        self.lire_stop = Gtk.Button(icon_name="media-playback-stop-symbolic")
        self.lire_stop.set_tooltip_text("Arrêter la lecture")
        self.lire_stop.connect("clicked", self.lire_arret)
        lire.append(self.lire_stop)
        box.append(lire)

        # ---- live translation (geshtu/traduction.py) ------------------
        self.traduire = Gtk.Button(label="🌐 Traduire en direct")
        self.traduire.set_tooltip_text(
            "La source choisie, de « Audio en » vers « Traduire vers » : "
            "sous-titres ici, voix Kokoro dans le casque")
        self.traduire.connect("clicked", self.traduire_bascule)
        box.append(self.traduire)
        self.sous_titres = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.sous_titres.set_left_margin(8)
        self.sous_titres.set_right_margin(8)
        haut = Gtk.ScrolledWindow(vexpand=True, child=self.sous_titres)
        haut.add_css_class("frame")
        box.append(haut)
        self.vu = 0

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
        GLib.timeout_add(1000, self.tick)

    # ------------------------------------------------------------- helpers
    def log(self, text: str) -> None:
        buf = self.view.get_buffer()
        buf.insert(buf.get_end_iter(), text.rstrip() + "\n")
        self.view.scroll_to_mark(
            buf.create_mark(None, buf.get_end_iter(), False), 0, False, 0, 0)

    def reload(self) -> None:
        keep = self.selected_key()
        r = call({"cmd": "targets"})
        # ⚠️ LE MICRO EST UNE LIGNE COMME LES AUTRES. Il a ete une case a
        # cocher separee ; ezvk : « checkbox also record my mic is useless as
        # your voice is reflected in main rendered/recorded stream ». Vrai
        # d un jeu ou d une capture de la sortie systeme. Dans une visio
        # navigateur en revanche le flux de l onglet porte la voix des AUTRES
        # -- la sienne part vers eux -- donc la source reste offerte, en un
        # clic, au lieu d etre une seconde question posee a part.
        #
        # ⚠️ ET LE CHOIX UNIQUE REND L ECHO IMPOSSIBLE, ce qui vaut mieux que
        # de le documenter. ezvk : « it may even make an echo ». Melanger le
        # micro a un flux qui porte deja la voix la double avec un leger
        # decalage : ce n est pas qu un desagrement a l oreille, l ASR repete
        # ou bafouille sur les passages doubles. On ne peut plus cocher les
        # deux, donc le cas ne se presente plus.
        self.rows = r.get("targets", [])
        textes = []
        for t in self.rows:
            if t["kind"] == "app":
                textes.append(("%s — %s" % (t["label"], t["detail"] or t["label"]))[:90])
            else:
                textes.append(("%s — %s" % (t["label"], t["detail"]))[:90])
        self.sources_menu.set_model(Gtk.StringList.new(
            textes or ["rien ne joue — un flux n'existe que pendant qu'il joue"]))
        for i, t in enumerate(self.rows):
            if t["key"] == keep:
                self.sources_menu.set_selected(i)
                return
        self.sources_menu.set_selected(0)

    def selected_key(self) -> str | None:
        if not self.rows:
            return None
        i = self.sources_menu.get_selected()
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
        r = call({"cmd": "start", "targets": [key],
                  "language": self.lang_code()})
        self.log(r.get("error") or ("recording %s" % r.get("session")))
        self.tick()

    def set_language(self, _drop, _param) -> None:
        r = call({"cmd": "set-language", "language": self.lang_code()})
        self.log(r.get("error") or ("summary in %s from now on" % r["language"]))

    def set_entree(self, _drop, _param) -> None:
        i = self.entree.get_selected()
        if not 0 <= i < len(self.entrees):
            return
        r = call({"cmd": "set-entree", "langue_entree": self.entrees[i]})
        self.log(r.get("error") or ("audio en %s from now on" % r["langue_entree"]))

    def set_cible(self, _drop, _param) -> None:
        i = self.cible.get_selected()
        if not 0 <= i < len(self.cibles):
            return
        r = call({"cmd": "set-cible", "langue_cible": self.cibles[i]})
        self.log(r.get("error") or ("traduction vers %s" % r["langue_cible"]))

    def traduire_bascule(self, _button) -> None:
        if self.traduction_active:
            call({"cmd": "traduire-stop"})
        else:
            key = self.selected_key()
            if key is None:
                self.log("choisis une source")
                return
            r = call({"cmd": "traduire", "target": key})
            if not r.get("ok"):
                self.log(r.get("error", "traduction impossible"))
            elif not r.get("voix"):
                self.log("sortie système : sous-titres seulement (la voix se "
                         "retraduirait elle-même en boucle)")
        self.tick()

    def lire_bascule(self, _button) -> None:
        r = call({"cmd": "lire"})
        if not r.get("ok"):
            self.log(r.get("error", "lecture impossible"))
        self.tick()

    def lire_arret(self, _button) -> None:
        call({"cmd": "lire-stop"})
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
        tr = r.get("traduction") or {}
        self.traduction_active = tr.get("etat") == "running"
        self.sources_menu.set_sensitive(not self.recording and not self.traduction_active)
        self.traduire.set_label("⏹ Arrêter la traduction" if self.traduction_active
                                else "🌐 Traduire en direct")
        self.traduire.set_sensitive(not self.recording and not self.busy)
        lignes = tr.get("lignes") or []
        if len(lignes) != self.vu or (lignes and not self.traduction_active and self.vu == 0):
            buf = self.sous_titres.get_buffer()
            # ⚠️ LA TRADUCTION SEULE. ezvk : « il met les deux langues dans le
            # transcript ». L'original ne revient que si la traduction a échoué,
            # pour ne pas perdre la phrase.
            buf.set_text("\n".join(
                "%s  %s" % (l["quand"], l["traduction"] or "(%s)" % l["original"])
                for l in lignes))
            self.sous_titres.scroll_to_mark(
                buf.create_mark(None, buf.get_end_iter(), False), 0, False, 0, 0)
            self.vu = len(lignes)
        lec = r.get("lecture") or {}
        etat = lec.get("etat", "idle")
        self.lire.set_label({
            "preparing": "… préparation de la lecture",
            "playing": "⏸ Pause",
            "paused": "▶ Reprendre",
        }.get(etat, "▶ Lire le dernier résumé"))
        self.lire.set_sensitive(etat != "preparing")
        self.lire_stop.set_sensitive(etat in ("playing", "paused", "preparing"))
        buf = self.view.get_buffer()
        known = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        for line in r.get("log", []):
            if line not in known:
                self.log(line)
        return False


def main() -> int:
    # ⚠️ FORCED, NOT JUST PREFERRED. `gtk-application-prefer-dark-theme` asks
    # the active theme for its dark variant -- it does not depend on a
    # desktop-wide dark-mode setting existing at all, so this is the one
    # switch that reliably gives geshtu a dark window regardless of what
    # utu's GTK theme is otherwise set to.
    settings = Gtk.Settings.get_default()
    if settings is not None:
        settings.set_property("gtk-application-prefer-dark-theme", True)
    app = Gtk.Application(application_id="org.geshtu.Window")
    quit_action = Gio.SimpleAction.new("quit", None)
    quit_action.connect("activate", lambda *_: app.quit())
    app.add_action(quit_action)
    app.set_accels_for_action("app.quit", ["<Control>q", "<Control>w"])
    app.connect("activate", lambda a: Window(a).present())
    return app.run(None)


if __name__ == "__main__":
    raise SystemExit(main())
