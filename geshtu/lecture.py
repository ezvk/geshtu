"""Read aloud: the latest summary, or any text -- the current selection first.

ezvk, 2026-09-27: « bouton lire le résumé à haute voix dans geshtu », « pour lire
le dernier résumé produit », « avec bouton play pause ».
ezvk, 2026-10-17: « une option dans geshtu pour lire le texte sélectionné : je
sélectionne un morceau de texte dans le browser ou ailleurs et kokoro le lit à
haute voix ».

The text goes to the `tts` engine (Kokoro on utu's NPU, OVMS /v3/audio/speech)
one passage at a time, and `pw-play` plays a passage while the next one is being
synthesised: the first words come a couple of seconds after the key, whatever
the length of the text. Pause is SIGSTOP on OUR pw-play process, resume is
SIGCONT -- PipeWire simply stops receiving samples; nothing else is touched.

⚠️ WHY NOT A BROWSER EXTENSION. Looked at on 2026-10-17: « Narrate » (Chrome),
« Custom TTS Reader », « Kokoro Reader », « Kokoro TTS web page reader »
(Firefox). All post OpenAI-shaped requests WITHOUT a `language` field, and
Kokoro then phonemises French with English rules (below). They also only see
the browser; the Wayland primary selection covers every window, and the daemon
already has the voice, the language and the pause.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import queue
import re
import signal
import struct
import subprocess
import threading

from geshtu import commandes
from geshtu import engines

# ⚠️ KOKORO PHONEMISES IN en-us WHEN NO LANGUAGE IS SENT, and French read with
# English rules comes out broken (ezvk, listening: « sans langue est cassé »).
# The text's own language decides; there is no server-side default to lean on.
# Phonemisation codes from docs/model_server_rest_api_text_to_speech.md, and one
# voice per language, the model's own (ff_siwis: its only French voice).
KOKORO = {"fr": ("fr-fr", "ff_siwis"), "en": ("en-us", "af_heart"),
          "es": ("es", "ef_dora"), "it": ("it", "if_sara"), "pt": ("pt-br", "pf_dora")}
LANGUES = {l: code for l, (code, _) in KOKORO.items()}
VOIX = {l: voix for l, (_, voix) in KOKORO.items()}

# Kokoro is happiest with short inputs; a paragraph at a time also bounds the
# cost of one failed request. The FIRST passage is shorter still: it is the one
# the listener waits for (measured 2026-09-27: ~0.13 s of synthesis per second
# of speech on the NPU, so 180 characters start in about two seconds).
MORCEAU = 450
PREMIER = 180

# ⚠️ A SELECTION HAS NO DECLARED LANGUAGE, unlike a summary. Function words
# are counted: they are frequent, short, and almost never shared between these
# five languages. Fewer than two hits, or a tie, falls back to the daemon's
# summary language -- a guess that reads French with an English voice is the
# very failure this file exists to avoid, so the fallback is the known choice.
MOTS = {
    "fr": set("le la les des une un et est pas que qui dans pour sur avec ce cette il elle "
              "nous vous sont au aux du ne se son sa ses mais ou où été être cela plus".split()),
    "en": set("the and is are of to in that it for with on this was as be by not have from "
              "you your they their which has will would can there".split()),
    "es": set("el los las una por con para que es del se su al como más pero sus han "
              "está esta este también sobre entre sin".split()),
    "it": set("il gli della delle che per con una sono del nel anche come più dei degli "
              "nella alla questo questa ma non è".split()),
    "pt": set("os as uma do da dos das que com para não por mais como seu sua são foi "
              "também pelo pela isso esse essa".split()),
}


def devine_langue(texte: str, defaut: str = "fr") -> str:
    mots = re.findall(r"[a-zà-ÿ]+", texte.lower())
    if len(mots) < 4:
        return defaut
    scores = {l: sum(m in s for m in mots) for l, s in MOTS.items()}
    haut = max(scores.values())
    gagnants = [l for l, n in scores.items() if n == haut]
    if haut < 2 or len(gagnants) > 1:
        return defaut
    return gagnants[0]


def dernier_resume(sessions: pathlib.Path) -> dict | None:
    """The session whose summary was produced LAST -- by the file's mtime, not
    by the session id: a reprocessed old session carries the newest summary."""
    best = None
    if not sessions.is_dir():
        return None
    for d in sessions.iterdir():
        f = d / "session.json"
        if not f.is_file():
            continue
        raw = json.loads(f.read_text())
        chapitres = [c for c in raw.get("chapters", []) if c.get("title")]
        if not (raw.get("summary") or chapitres):
            continue
        m = f.stat().st_mtime
        if best is None or m > best[0]:
            best = (m, d, raw, chapitres)
    if best is None:
        return None
    _, d, raw, chapitres = best
    return {"root": d, "id": raw["id"], "title": raw.get("title", ""),
            "language": raw.get("language", "fr"), "cache": True,
            "texte": texte_parle(raw.get("title", ""), raw.get("summary", ""), chapitres)}


def texte_parle(titre: str, resume: str, chapitres: list[dict]) -> str:
    """What the note says, without what only a reader of Markdown needs."""
    parts = [titre] if titre else []
    parts.append(resume)
    for c in chapitres:
        parts += [c["title"], c.get("body", "")]
    return texte_libre("\n\n".join(p for p in parts if p))


def texte_libre(t: str) -> str:
    """Any text -- a selection, a note -- with what cannot be spoken taken out."""
    t = re.sub(r"https?://[^\s)\]>]+", "", t)     # a URL read aloud is noise
    t = re.sub(r"`[^`\n]*`", "", t)                 # timestamps, code
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # links -> their text
    t = re.sub(r"[*_#>|]+", "", t)                  # emphasis, headings, quotes
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)   # bullets
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # hyphenation at a line break
    return re.sub(r"[ \t]+", " ", t).strip()


def morceaux(texte: str, premier: int = PREMIER) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", texte):
        para = para.replace("\n", " ").strip()
        while para:
            lim = premier if not out else MORCEAU
            if len(para) <= lim:
                out.append(para)
                break
            cut = max(para.rfind(". ", 0, lim), para.rfind("? ", 0, lim),
                      para.rfind("! ", 0, lim))
            if cut <= 0:
                cut = max(para.rfind(", ", 0, lim), para.rfind("; ", 0, lim))
            if cut <= 0:
                cut = para.rfind(" ", 0, lim)
            cut = cut + 1 if cut > 0 else lim
            out.append(para[:cut].strip())
            para = para[cut:].strip()
    return out


def selection_courante() -> tuple[str, str]:
    """(text, where it came from): the primary selection -- what is highlighted
    right now, in any Wayland window -- else the clipboard.

    ⚠️ `wl-paste` NEEDS THE DISPLAY. The daemon runs as a systemd --user
    service; on utu it inherits WAYLAND_DISPLAY from the session (checked
    2026-10-17), and horde puts wl-clipboard in its PATH. Anything else -- an
    image, a file -- is not text and is skipped.
    """
    for args, nom in ((["--primary"], "sélection"), ([], "presse-papier")):
        try:
            r = subprocess.run(["wl-paste", "--no-newline", *args],
                               capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("wl-paste: %s" % exc)
        if r.returncode != 0 or not r.stdout.strip():
            continue
        try:
            return r.stdout.decode().strip(), nom
        except UnicodeDecodeError:
            continue
    return "", ""


def cible_texte(texte: str, defaut: str) -> dict:
    propre = texte_libre(texte)
    base = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "geshtu-lecture"
    base.mkdir(parents=True, exist_ok=True)
    titre = propre.split("\n", 1)[0][:70] + ("…" if len(propre) > 70 else "")
    return {"root": base, "id": "texte", "title": titre, "cache": False,
            "language": devine_langue(propre, defaut), "texte": propre}


class Lecteur:
    """One reading at a time. States: idle, preparing, playing, paused."""

    def __init__(self, state) -> None:
        self.state = state                      # daemon State: cfg, report()
        self.lock = threading.Lock()
        self.etat = "idle"
        self.session = ""
        self.titre = ""
        self.brut: str | None = None            # the text as handed in, to tell a new one
        self.proc: subprocess.Popen | None = None
        # ⚠️ A READING IS ORPHANED, NOT JOINED. A new selection while one plays
        # must start at once; the old thread finishes its current pw-play and
        # finds its token stale, so it touches nothing. Joining it would wait
        # for Kokoro to answer the passage in flight.
        self.jeton = 0
        self.arret = threading.Event()

    def status(self) -> dict:
        with self.lock:
            return {"etat": self.etat, "session": self.session, "titre": self.titre}

    def bascule(self, texte: str | None = None, selection: bool = False) -> dict:
        """No text: play/pause the latest summary. A text (or `selection`):
        read it; the same text again pauses or resumes; another text replaces
        what is playing."""
        provenance = ""
        if selection:
            try:
                texte, provenance = selection_courante()
            except RuntimeError as exc:
                return {"ok": False, "error": str(exc)}
            if not texte:
                commandes.dire("Rien à lire", "aucun texte sélectionné ni copié")
                return {"ok": False, "error": "nothing selected, nothing copied"}
        with self.lock:
            if self.etat != "idle" and (texte is None or texte == self.brut):
                if self.etat == "preparing":
                    return {"ok": True, "etat": self.etat}
                if self.proc:
                    try:
                        os.kill(self.proc.pid,
                                signal.SIGSTOP if self.etat == "playing" else signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                self.etat = "paused" if self.etat == "playing" else "playing"
                return {"ok": True, "etat": self.etat}
            if self.etat != "idle":
                self._arrete_verrouille()           # another text: it replaces this one
            if texte is None:
                cible = dernier_resume(self.state.cfg.sessions)
                if cible is None:
                    return {"ok": False, "error": "no summary has been produced yet"}
            else:
                cible = cible_texte(texte, self.state.language())
                if not cible["texte"]:
                    return {"ok": False, "error": "nothing to read in that text"}
            self.jeton += 1
            jeton = self.jeton
            self.arret = threading.Event()
            self.etat, self.session, self.titre = "preparing", cible["id"], cible["title"]
            self.brut = texte
        if texte is not None:
            commandes.dire("Lecture (%s, %s)" % (provenance or "texte", cible["language"]),
                           cible["title"])
        threading.Thread(target=self._lire, args=(cible, jeton), daemon=True).start()
        return {"ok": True, "etat": "preparing", "session": cible["id"],
                "langue": cible["language"], "titre": cible["title"]}

    def stop(self) -> dict:
        with self.lock:
            self._arrete_verrouille()
        return {"ok": True, "etat": "idle"}

    def _arrete_verrouille(self) -> None:
        """Caller holds `self.lock`."""
        self.arret.set()
        self.jeton += 1
        p = self.proc
        if p and p.poll() is None:
            # A stopped process does not act on SIGTERM until it runs again.
            try:
                os.kill(p.pid, signal.SIGCONT)
                p.terminate()
            except ProcessLookupError:
                pass
        self.etat, self.proc, self.brut = "idle", None, None

    # -- one reading, passage by passage ---------------------------------
    def _voix(self, cible: dict) -> tuple[str, str]:
        lang = cible["language"] if cible["language"] in KOKORO else "fr"
        code, voix = KOKORO[lang]
        voix = self.state.cfg.raw.get("lecture", {}).get("voix_" + lang) or voix
        return code, voix

    def _cache(self, cible: dict, voix: str, lang: str) -> pathlib.Path | None:
        """A summary's audio is kept beside its session; a selection is not kept.
        ⚠️ THE CACHE KEY COVERS WHAT CHANGES THE SOUND: text, voice, language,
        model. A reprocessed summary, or another voice, must not replay old audio."""
        if not cible.get("cache"):
            return None
        eng = self.state.cfg.engine("tts")
        cle = hashlib.sha256("\0".join(
            [cible["texte"], voix, lang, eng.endpoint, eng.model]).encode()).hexdigest()[:16]
        return cible["root"] / ("resume-%s.wav" % cle)

    def _lire(self, cible: dict, jeton: int) -> None:
        arret = self.arret
        try:
            lang, voix = self._voix(cible)
            cache = self._cache(cible, voix, lang)
            if cache is not None and cache.is_file():
                self.state.report("reading aloud %s (cached)" % cible["id"])
                self._joue(cache, jeton, arret)
                return
            parts = morceaux(cible["texte"])
            self.state.report("reading aloud %s: %d passages (%s, %s)"
                              % (cible["id"], len(parts), voix, lang))
            eng = self.state.cfg.engine("tts")
            file: queue.Queue = queue.Queue(maxsize=2)

            def produit() -> None:
                for p in parts:
                    if arret.is_set():
                        break
                    try:
                        file.put((_riff(engines.speak(eng, p, voix, lang)), None))
                    except Exception as exc:                 # noqa: BLE001
                        file.put((None, exc))
                        break
                file.put(None)

            threading.Thread(target=produit, daemon=True).start()
            fmt0, donnees, i, complet = None, [], 0, True
            while True:
                item = file.get()
                if item is None:
                    break
                paquet, exc = item
                if exc is not None:
                    self.state.report("read aloud: kokoro failed: %s" % exc)
                    commandes.dire("Lecture interrompue", str(exc)[:80])
                    complet = False
                    break
                fmt, data = paquet
                if fmt0 is None:
                    fmt0 = fmt
                elif fmt != fmt0:
                    complet = False                          # joinable only if alike
                if arret.is_set():
                    break
                f = cible["root"] / ("lecture-%d.wav" % (i % 2))
                i += 1
                f.write_bytes(_wav(fmt, data))
                donnees.append(data)
                if not self._joue(f, jeton, arret):
                    break
            if (cache is not None and complet and fmt0 is not None
                    and not arret.is_set() and len(donnees) == len(parts)):
                for vieux in cible["root"].glob("resume-*.wav"):
                    vieux.unlink()
                tmp = cache.with_suffix(".tmp")
                tmp.write_bytes(_wav(fmt0, b"".join(donnees)))
                tmp.replace(cache)
        except Exception as exc:                             # noqa: BLE001
            self.state.report("read aloud failed: %s" % exc)
        finally:
            with self.lock:
                if self.jeton == jeton:
                    self.etat, self.proc, self.brut = "idle", None, None

    def _joue(self, f: pathlib.Path, jeton: int, arret: threading.Event) -> bool:
        """Play one file to the end. False when this reading was stopped."""
        with self.lock:
            if self.jeton != jeton or arret.is_set():
                return False
            self.proc = subprocess.Popen(["pw-play", str(f)])
            p = self.proc
            if self.etat == "preparing":
                self.etat = "playing"
            elif self.etat == "paused":
                # Paused exactly between two passages: the next one waits too.
                os.kill(p.pid, signal.SIGSTOP)
        p.wait()
        return not arret.is_set()


# ⚠️ NOT THE `wave` MODULE: Kokoro answers 32-bit FLOAT WAV (format tag 3), and
# `wave` only reads integer PCM -- « wave.Error: unknown format: 3 ». Joining
# passages needs nothing more than the `fmt ` chunk of the first one and the
# `data` chunks of all of them, so it is done by hand.

def _riff(blob: bytes) -> tuple[bytes, bytes]:
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise RuntimeError("not a WAV answer: %r" % blob[:60])
    fmt = data = None
    i = 12
    while i + 8 <= len(blob):
        cid, size = blob[i:i + 4], struct.unpack("<I", blob[i + 4:i + 8])[0]
        body = blob[i + 8:i + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        i += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise RuntimeError("WAV answer without fmt/data chunk")
    return fmt, data


def _wav(fmt: bytes, data: bytes) -> bytes:
    corps = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
             + b"data" + struct.pack("<I", len(data)) + data)
    return b"RIFF" + struct.pack("<I", len(corps)) + corps
