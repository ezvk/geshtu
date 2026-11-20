"""The shared screen: keep one picture each time it changes and then holds still.

A slide, a step of a demonstration, a table someone scrolls to: what matters on
a shared screen is each state that was SHOWN, not a film of it. So the screen is
sampled once a second in a small grey thumbnail, and a full picture is kept when
it has changed since the last one kept and has then been still for a moment.
Measured on a 15 min conference (2026-11-20): one picture per slide, 29 in all.

⚠️ A VIDEO TILE NEVER HOLDS STILL. A call shows the speaker's camera beside the
slides; counted with the rest, that tile is "movement" every second and nothing
is ever kept. Each cell of a coarse grid therefore remembers how often it moved
lately: a cell that moves most of the time is a living picture and is left out
of both questions (did it change? is it still?). A slide change touches far
more than the cells of a camera tile.

No dependency: thumbnails are a few thousand pixels, plain Python is enough.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import re
import subprocess
import threading
import time

GRILLE = (8, 6)          # cells across, down


@dataclasses.dataclass
class Reglages:
    pixel: int = 25            # grey-level difference that counts as a changed pixel
    cellule: float = 0.03      # share of ONE cell's pixels changed since the last picture kept
    mouvement: float = 0.003   # share of watched pixels moving from one second to the next
    stable: int = 2            # seconds the screen must hold still
    vivant: int = 5            # a cell that moved ON ITS OWN this often in the last minute is video

    @classmethod
    def depuis(cls, raw: dict) -> "Reglages":
        champs = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in champs})


class Detecteur:
    """Fed one grey thumbnail per second (bytes, w, h); answers True when the
    current picture is one to keep."""

    def __init__(self, reglages: Reglages | None = None):
        self.r = reglages or Reglages()
        self.ref = None
        self.prev = None
        self.attente = True
        self.calme = 0
        self.histoire: dict[int, collections.deque] = {}

    def _cellule(self, i: int, w: int, h: int) -> int:
        x, y = i % w, i // w
        return (y * GRILLE[1] // h) * GRILLE[0] + (x * GRILLE[0] // w)

    def voit(self, gris: bytes, w: int, h: int) -> bool:
        if self.prev is None or len(self.prev) != len(gris):
            # First picture, or the size changed (another screen): start over.
            self.prev = self.ref = gris
            self.attente, self.calme = True, 0
            self.histoire = {c: collections.deque(maxlen=60)
                             for c in range(GRILLE[0] * GRILLE[1])}
            return False
        p = self.r.pixel
        bouge_cell = collections.Counter()
        ecart_cell = collections.Counter()
        taille = collections.Counter()
        for i in range(len(gris)):
            c = self._cellule(i, w, h)
            taille[c] += 1
            g = gris[i]
            if abs(g - self.prev[i]) > p:
                bouge_cell[c] += 1
            if abs(g - self.ref[i]) > p:
                ecart_cell[c] += 1
        # ⚠️ ONLY A LOCAL MOVEMENT MAKES A CELL "VIVANT". A slide change moves
        # most cells at once, and a slide shown every 20 s would otherwise mark
        # the slide itself as video and be ignored from the third one on. A
        # camera tile moves on its own: those are the movements counted.
        bouges = {c for c in self.histoire if bouge_cell[c] > 0.02 * max(taille[c], 1)}
        local = len(bouges) < len(self.histoire) // 4
        for c, hist in self.histoire.items():
            hist.append(local and c in bouges)
        vivants = {c for c, hist in self.histoire.items() if sum(hist) >= self.r.vivant}
        regardes = sum(n for c, n in taille.items() if c not in vivants) or 1
        bouge = sum(n for c, n in bouge_cell.items() if c not in vivants) / regardes
        # ⚠️ A CHANGE IS JUDGED CELL BY CELL, not over the whole screen. A bullet
        # revealed on a slide, a scatter of small dots, a menu opening: a few
        # hundred pixels of a 14 000-pixel thumbnail, under any global share
        # that noise does not also cross -- and exactly the content the notes
        # need (measured: 4 of 29 slide states lost with a global 1 %).
        change = any(n > self.r.cellule * taille[c]
                     for c, n in ecart_cell.items() if c not in vivants)
        self.prev = gris
        if change:
            self.attente = True
        self.calme = self.calme + 1 if bouge < self.r.mouvement else 0
        if self.attente and self.calme >= self.r.stable:
            self.ref, self.attente = gris, False
            return True
        return False


# -- a video file --------------------------------------------------------------

def dans_video(video: pathlib.Path, dossier: pathlib.Path, reglages: Reglages,
               report=print) -> list[tuple[float, pathlib.Path]]:
    """The screens of a media file, as (seconds, picture) pairs."""
    w, h = 160, 90
    brut = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf",
         "fps=1,scale=%d:%d,format=gray" % (w, h), "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    det = Detecteur(reglages)
    instants = [t for t in range(len(brut) // (w * h))
                if det.voit(brut[t * w * h:(t + 1) * w * h], w, h)]
    dossier.mkdir(parents=True, exist_ok=True)
    out = []
    for k, t in enumerate(instants):
        image = dossier / ("e%03d-%s.jpg" % (k, _hms(t)))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(video),
                        "-frames:v", "1", "-q:v", "2", str(image)], check=True)
        out.append((float(t), image))
    report("screens: %d kept in %s" % (len(out), video.name))
    return out


def a_des_images(media: pathlib.Path) -> bool:
    """True when the file carries a real video stream (not a cover picture)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "stream=codec_name:stream_disposition=attached_pic",
                        "-of", "csv=p=0", str(media)], capture_output=True, text=True)
    return any(ligne and not ligne.endswith(",1") for ligne in r.stdout.splitlines())


# -- the live screen -------------------------------------------------------------

class Capture:
    """Watches the screen during a recording (grim, wlroots screencopy) and
    appends each kept state to session.ecrans.

    ⚠️ IT SEES THE WHOLE SCREEN, NOTIFICATIONS INCLUDED. Off unless [ecran]
    actif = true, and `sortie` / `zone` narrow it to one output or one region.

    `fenetre` (an app-id, or a piece of the title) watches ONE window instead,
    through ext-foreign-toplevel-list + ext-image-copy-capture (grim -T). Measured
    on utu under mango (2026-11-20): the window is captured whole and up to date
    while it sits on a hidden tag, AND while the screen is asleep -- which the
    whole-screen capture is not (grim timed out on every try, screen off). The
    visio can then stay on its own tag while work goes on elsewhere (ezvk).
    """

    def __init__(self, session, raw: dict, report=print):
        self.session = session
        self.raw = raw or {}
        self.report = report
        self.r = Reglages.depuis(self.raw)
        self.dossier = session.root / "ecrans"
        self.arret = threading.Event()
        self.fil: threading.Thread | None = None
        self.toplevel = ""

    def _grim(self, *options: str) -> list[str]:
        cmd = ["grim"]
        if self.raw.get("fenetre"):
            # Looked up again whenever lost: a window closed and reopened (a
            # meeting popup) comes back under a new identifier.
            if not self.toplevel:
                self.toplevel = fenetre(self.raw["fenetre"])
            if not self.toplevel:
                raise RuntimeError("no window matches %r" % self.raw["fenetre"])
            return cmd + ["-T", self.toplevel] + list(options)
        if self.raw.get("sortie"):
            cmd += ["-o", self.raw["sortie"]]
        if self.raw.get("zone"):
            cmd += ["-g", self.raw["zone"]]
        return cmd + list(options)

    def start(self) -> None:
        self.dossier.mkdir(parents=True, exist_ok=True)
        self.fil = threading.Thread(target=self._boucle, daemon=True)
        self.fil.start()

    def stop(self) -> None:
        self.arret.set()
        if self.fil:
            self.fil.join(timeout=5)

    def _boucle(self) -> None:
        det = Detecteur(self.r)
        echecs = 0
        while not self.arret.wait(1.0):
            try:
                gris, w, h = _vignette(self._grim("-s", "0.1", "-t", "ppm", "-"))
            except Exception as exc:                     # noqa: BLE001
                self.toplevel = ""
                echecs += 1
                if echecs in (1, 30):      # said once, and again if it persists
                    self.report("screen capture failed: %s" % exc)
                continue
            echecs = 0
            if not det.voit(gris, w, h):
                continue
            t = time.time() - self.session.started
            image = self.dossier / ("e%03d-%s.jpg" % (len(self.session.ecrans), _hms(t)))
            r = subprocess.run(self._grim("-t", "jpeg", "-q", "85", str(image)),
                               capture_output=True)
            if r.returncode == 0:
                self.session.ecrans.append(_ecran(t, image))
                self.report("screen kept at %s" % _hms(t))


def fenetre(motif: str) -> str:
    """The foreign-toplevel identifier of the window `motif` names, "" when
    none is open. Read with lswt (ext-foreign-toplevel-list-v1). In order:
    the app-id exactly; the app-id spelled like the audio stream's program
    (« Teams for Linux » = teams-for-linux, « Chromium » ⊂ chromium-browser),
    a meeting title first when the program has several windows; a title.
    """
    r = subprocess.run(["lswt", "-j"], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError("lswt: %s" % r.stderr.strip()[:200])
    fenetres = json.loads(r.stdout).get("toplevels", [])
    for f in fenetres:
        if f.get("app-id") == motif:
            return f.get("identifier", "")
    cle = _plat(motif)
    siennes = [f for f in fenetres
               if cle and (cle in _plat(f.get("app-id")) or _plat(f.get("app-id")) in cle)
               and _plat(f.get("app-id"))]
    for f in siennes:
        if any(m in (f.get("title") or "").lower() for m in REUNIONS):
            return f.get("identifier", "")
    if siennes:
        return siennes[0].get("identifier", "")
    for f in fenetres:
        if motif.lower() in (f.get("title") or "").lower():
            return f.get("identifier", "")
    return ""


# Words a meeting window carries in its title, to pick it among a browser's
# windows.
REUNIONS = ("meet", "teams", "zoom", "jitsi", "webex", "whereby", "visio", "réunion")


def _plat(texte) -> str:
    return re.sub(r"[^a-z0-9]", "", (texte or "").lower())


def _ecran(t, image):
    from geshtu.models import Ecran
    return Ecran(t=t, image=image)


def _vignette(cmd: list[str]) -> tuple[bytes, int, int]:
    """A grim PPM (P6) from stdout, as grey bytes."""
    r = subprocess.run(cmd, capture_output=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip() or "grim failed")
    data = r.stdout
    champs, pos = [], 2
    if data[:2] != b"P6":
        raise RuntimeError("not a PPM picture")
    while len(champs) < 3:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        fin = pos
        while not data[fin:fin + 1].isspace():
            fin += 1
        champs.append(int(data[pos:fin]))
        pos = fin
    w, h, _ = champs
    rgb = data[pos + 1:pos + 1 + w * h * 3]
    gris = bytes((rgb[i] * 299 + rgb[i + 1] * 587 + rgb[i + 2] * 114) // 1000
                 for i in range(0, len(rgb), 3))
    return gris, w, h


def _hms(seconds: float) -> str:
    s = int(seconds)
    return "%02dh%02dm%02ds" % (s // 3600, (s % 3600) // 60, s % 60)
