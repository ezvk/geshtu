"""Read the latest summary aloud, with play / pause / stop.

ezvk, 2026-09-27: « bouton lire le résumé à haute voix dans geshtu », « pour lire
le dernier résumé produit », « avec bouton play pause ».

The text goes to the `tts` engine (Kokoro on utu's NPU, OVMS /v3/audio/speech),
the audio is cached beside the session, and `pw-play` plays it. Pause is
SIGSTOP on OUR pw-play process, resume is SIGCONT -- PipeWire simply stops
receiving samples; nothing else is touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import struct
import threading

from geshtu import engines

# ⚠️ KOKORO PHONEMISES IN en-us WHEN NO LANGUAGE IS SENT, and French read with
# English rules comes out broken (ezvk, listening: « sans langue est cassé »).
# The summary's own language decides; there is no server-side default to lean on.
LANGUES = {"fr": "fr-fr", "en": "en-us"}
VOIX = {"fr": "ff_siwis", "en": "af_heart"}   # ff_siwis: the model's only French voice

# Kokoro is happiest with short inputs; a paragraph at a time also bounds the
# cost of one failed request.
MORCEAU = 450


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
            "language": raw.get("language", "fr"),
            "texte": texte_parle(raw.get("title", ""), raw.get("summary", ""), chapitres)}


def texte_parle(titre: str, resume: str, chapitres: list[dict]) -> str:
    """What the note says, without what only a reader of Markdown needs."""
    parts = [titre] if titre else []
    parts.append(resume)
    for c in chapitres:
        parts += [c["title"], c.get("body", "")]
    t = "\n\n".join(p for p in parts if p)
    t = re.sub(r"`[^`]*`", "", t)                 # timestamps, code
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # links -> their text
    t = re.sub(r"[*_#>|]+", "", t)                # emphasis, headings, quotes
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)  # bullets
    return re.sub(r"[ \t]+", " ", t).strip()


def morceaux(texte: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", texte):
        para = para.replace("\n", " ").strip()
        while len(para) > MORCEAU:
            cut = max(para.rfind(". ", 0, MORCEAU), para.rfind(", ", 0, MORCEAU))
            cut = cut + 1 if cut > 0 else MORCEAU
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            out.append(para)
    return out


class Lecteur:
    """One reading at a time. States: idle, preparing, playing, paused."""

    def __init__(self, state) -> None:
        self.state = state                      # daemon State: cfg, report()
        self.lock = threading.Lock()
        self.etat = "idle"
        self.session = ""
        self.titre = ""
        self.proc: subprocess.Popen | None = None

    def status(self) -> dict:
        with self.lock:
            return {"etat": self.etat, "session": self.session, "titre": self.titre}

    def bascule(self) -> dict:
        with self.lock:
            if self.etat == "playing" and self.proc:
                os.kill(self.proc.pid, signal.SIGSTOP)
                self.etat = "paused"
                return {"ok": True, "etat": self.etat}
            if self.etat == "paused" and self.proc:
                os.kill(self.proc.pid, signal.SIGCONT)
                self.etat = "playing"
                return {"ok": True, "etat": self.etat}
            if self.etat == "preparing":
                return {"ok": True, "etat": self.etat}
            cible = dernier_resume(self.state.cfg.sessions)
            if cible is None:
                return {"ok": False, "error": "no summary has been produced yet"}
            self.etat, self.session, self.titre = "preparing", cible["id"], cible["title"]
        threading.Thread(target=self._lire, args=(cible,), daemon=True).start()
        return {"ok": True, "etat": "preparing", "session": cible["id"]}

    def stop(self) -> dict:
        with self.lock:
            p = self.proc
            if p and p.poll() is None:
                # A stopped process does not act on SIGTERM until it runs again.
                os.kill(p.pid, signal.SIGCONT)
                p.terminate()
            self.etat, self.proc = "idle", None
        return {"ok": True, "etat": "idle"}

    def _lire(self, cible: dict) -> None:
        try:
            wav = self._synthese(cible)
        except Exception as exc:                   # noqa: BLE001
            self.state.report("read aloud failed: %s" % exc)
            with self.lock:
                self.etat = "idle"
            return
        with self.lock:
            if self.etat != "preparing":           # stopped while preparing
                return
            self.proc = subprocess.Popen(["pw-play", str(wav)])
            self.etat = "playing"
            p = self.proc
        self.state.report("reading aloud %s" % cible["id"])
        p.wait()
        with self.lock:
            if self.proc is p:
                self.etat, self.proc = "idle", None

    def _synthese(self, cible: dict) -> pathlib.Path:
        cfg = self.state.cfg
        eng = cfg.engine("tts")
        lang = cible["language"] if cible["language"] in LANGUES else "fr"
        opts = cfg.raw.get("lecture", {})
        voix = opts.get("voix_" + lang) or VOIX[lang]
        # ⚠️ THE CACHE KEY COVERS WHAT CHANGES THE SOUND: text, voice, language,
        # model. A reprocessed summary, or another voice, must not replay old audio.
        cle = hashlib.sha256("\0".join(
            [cible["texte"], voix, lang, eng.endpoint, eng.model]).encode()).hexdigest()[:16]
        out = cible["root"] / ("resume-%s.wav" % cle)
        if out.is_file():
            return out
        for vieux in cible["root"].glob("resume-*.wav"):
            vieux.unlink()
        parts = morceaux(cible["texte"])
        self.state.report("synthesising %d passages (%s, %s)" % (len(parts), voix, LANGUES[lang]))
        fmt, donnees = None, []
        for p in parts:
            f, d = _riff(engines.speak(eng, p, voix, LANGUES[lang]))
            if fmt is None:
                fmt = f
            elif f != fmt:
                raise RuntimeError("the passages came back in different audio formats")
            donnees.append(d)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(_wav(fmt, b"".join(donnees)))
        tmp.replace(out)
        return out


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
