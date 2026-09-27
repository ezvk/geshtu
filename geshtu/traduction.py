"""Live translation of the source picked in the window: Whisper -> LLM -> Kokoro.

ezvk, 2026-09-27: « real time translation ? », then « on traduit le sélecteur
d'audio de geshtu, voix koko et fenêtre gesh ».

NOT word by word: neither Whisper nor Kokoro stream in OVMS. The capture file
that `pw-record` is writing is read as it grows, cut on PAUSES, and each
phrase goes through three calls -- measured 2026-09-27: Whisper 0.4-0.6 s on
the NPU, `reflexe` (Gemma, no thinking) 0.4-1 s on ishtar, Kokoro ~0.13 s per
second of speech on the NPU. The translation therefore lands about two
seconds after the end of each phrase, like an interpreter waiting for the
full stop.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import queue
import struct
import subprocess
import threading
import time
import wave

from geshtu import engines, models
from geshtu.lecture import _riff, _wav

# Kokoro's phonemisation codes (docs/model_server_rest_api_text_to_speech.md)
# and one voice per language, the model's own.
KOKORO = {"fr": ("fr-fr", "ff_siwis"), "en": ("en-us", "af_heart"),
          "es": ("es", "ef_dora"), "it": ("it", "if_sara"), "pt": ("pt-br", "pf_dora")}
NOMS = {"fr": "français", "en": "anglais", "es": "espagnol", "it": "italien",
        "pt": "portugais"}

RATE = 16000
TRAME = 480                 # 30 ms at 16 kHz
SEUIL_VOIX = 700            # int16 RMS, ~ -33 dBFS: below is a pause
PAUSE = 0.6                 # seconds of pause that end a phrase
MIN_VOIX = 0.8              # a phrase needs at least this much speech
MAX_PHRASE = 12.0           # cut anyway: nobody waits longer than this

# Whisper's known inventions on near-silence. A clip that says only this is
# dropped rather than translated and spoken.
FANTOMES = {"merci.", "thank you.", "thanks for watching!", "sous-titrage st' 501",
            "sous-titres réalisés par la communauté d'amara.org", "you", "."}


class Traducteur:
    """One live translation at a time. States: idle, running."""

    def __init__(self, state) -> None:
        self.state = state
        self.lock = threading.Lock()
        self.etat = "idle"
        self.source = ""
        self.cible = ""
        self.voix = False
        self.lignes: list[dict] = []
        self.session = None
        self.src = None
        self.arret = threading.Event()
        self.a_dire: queue.Queue = queue.Queue()

    def status(self) -> dict:
        with self.lock:
            return {"etat": self.etat, "source": self.source, "cible": self.cible,
                    "voix": self.voix, "lignes": self.lignes[-40:]}

    # -- start / stop -------------------------------------------------------
    def start(self, key: str, entree: str, cible: str) -> dict:
        if cible not in KOKORO:
            return {"ok": False, "error": "no Kokoro voice for %r (have: %s)"
                    % (cible, ", ".join(KOKORO))}
        if entree == cible:
            return {"ok": False, "error": "the audio is already in %s" % NOMS.get(cible, cible)}
        with self.lock:
            if self.etat != "idle":
                return {"ok": False, "error": "already translating"}
            if self.state.session is not None:
                return {"ok": False, "error": "a recording is running -- one capture at a time"}
            src = self.state.source()
            known = {t.key: t for t in src.targets()}
            target = known.get(key)
            if target is None:
                return {"ok": False, "error": "unknown source %r -- refresh the list" % key}
            root = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "geshtu-traduction"
            root.mkdir(parents=True, exist_ok=True)
            for vieux in root.iterdir():
                vieux.unlink()
            session = models.Session(id="traduction", root=root)
            try:
                track = src.start(session, target, 0)
            except Exception as exc:                   # noqa: BLE001
                try:
                    src.stop(session)
                except Exception:                      # noqa: BLE001
                    pass
                return {"ok": False, "error": str(exc)}
            # ⚠️ NO VOICE WHEN THE SOURCE IS THE SYSTEM OUTPUT: Kokoro plays on
            # that very output, the capture would hear it and translate it
            # again, in a loop. An application stream is captured on its own
            # sink and does not hear Kokoro. Subtitles only, and said so.
            self.voix = target.kind not in ("output",) and key != "default:output"
            self.session, self.src = session, src
            self.etat, self.source, self.cible = "running", entree, cible
            self.lignes = []
            self.arret.clear()
            self.a_dire = queue.Queue()
        threading.Thread(target=self._ecoute, args=(track.segments[0], entree, cible),
                         daemon=True).start()
        if self.voix:
            threading.Thread(target=self._parle, daemon=True).start()
        self.state.report("live translation %s -> %s from %s%s" % (
            entree, cible, target.label, "" if self.voix else " (subtitles only: system output)"))
        return {"ok": True, "voix": self.voix}

    def stop(self) -> dict:
        with self.lock:
            if self.etat == "idle":
                return {"ok": True, "etat": "idle"}
            self.arret.set()
            session, src = self.session, self.src
            self.etat, self.session, self.src = "idle", None, None
        try:
            src.stop(session)
        except Exception as exc:                       # noqa: BLE001
            self.state.report("stopping the capture: %s" % exc)
        self.a_dire.put(None)
        # The capture grows at 32 kB/s in XDG_RUNTIME_DIR (RAM): drop it.
        for f in session.root.iterdir():
            if f.suffix == ".wav":
                f.unlink(missing_ok=True)
        self.state.report("live translation stopped")
        return {"ok": True, "etat": "idle"}

    # -- the capture, read while it grows ---------------------------------
    def _ecoute(self, wav: pathlib.Path, entree: str, cible: str) -> None:
        debut = None
        while debut is None and not self.arret.is_set():
            try:
                tete = wav.read_bytes()[:4096]
                i = tete.find(b"data")
                if i >= 0:
                    debut = i + 8
            except OSError:
                pass
            time.sleep(0.2)
        pos, phrase, voix, pause = debut or 0, bytearray(), 0.0, 0.0
        while not self.arret.is_set():
            try:
                with wav.open("rb") as f:
                    f.seek(pos)
                    neuf = f.read()
            except OSError:
                break
            neuf = neuf[: len(neuf) - len(neuf) % (TRAME * 2)]
            pos += len(neuf)
            for k in range(0, len(neuf), TRAME * 2):
                trame = neuf[k:k + TRAME * 2]
                n = len(trame) // 2
                ech = struct.unpack("<%dh" % n, trame)
                rms = (sum(x * x for x in ech) / n) ** 0.5
                if rms >= SEUIL_VOIX:
                    voix += TRAME / RATE
                    pause = 0.0
                    phrase += trame
                elif phrase:
                    pause += TRAME / RATE
                    phrase += trame
                duree = len(phrase) / 2 / RATE
                if phrase and ((pause >= PAUSE and voix >= MIN_VOIX) or duree >= MAX_PHRASE):
                    self._phrase(bytes(phrase), entree, cible)
                    phrase, voix, pause = bytearray(), 0.0, 0.0
                elif phrase and pause >= PAUSE:           # a cough, a click: drop it
                    phrase, voix, pause = bytearray(), 0.0, 0.0
            time.sleep(0.15)

    def _phrase(self, pcm: bytes, entree: str, cible: str) -> None:
        cfg = self.state.cfg
        clip = self.session.root / "phrase.wav" if self.session else None
        if clip is None:
            return
        with wave.open(str(clip), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        t0 = time.time()
        try:
            texte = engines.transcribe(cfg.engine("asr"), clip, entree).strip()
        except Exception as exc:                       # noqa: BLE001
            self.state.report("translation: whisper failed: %s" % exc)
            return
        if not texte or texte.lower() in FANTOMES:
            return
        try:
            trad = self._traduit(texte, entree, cible)
        except Exception as exc:                       # noqa: BLE001
            self.state.report("translation: the LLM failed: %s" % exc)
            trad = ""
        ligne = {"quand": time.strftime("%H:%M:%S"), "original": texte,
                 "traduction": trad, "delai": round(time.time() - t0, 1)}
        with self.lock:
            self.lignes.append(ligne)
        if trad and self.voix:
            self.a_dire.put(trad)

    def _traduit(self, texte: str, entree: str, cible: str) -> str:
        cfg = self.state.cfg
        eng = cfg.engine("llm")
        modele = cfg.raw.get("traduction", {}).get("model")
        if modele:
            eng = dataclasses.replace(eng, model=modele)
        consigne = ("Tu es un interprète. Traduis fidèlement ce texte parlé du %s "
                    "vers le %s. Réponds UNIQUEMENT par la traduction, sans guillemets, "
                    "sans commentaire, sans note." % (NOMS.get(entree, entree), NOMS[cible]))
        return engines.chat(eng, texte, system=consigne, max_tokens=300,
                            allow_truncated=True).strip()

    # -- the voice, one phrase after the other ----------------------------
    def _parle(self) -> None:
        cfg = self.state.cfg
        lang, voix = KOKORO[self.cible]
        voix = cfg.raw.get("lecture", {}).get("voix_" + self.cible) or voix
        n = 0
        while True:
            trad = self.a_dire.get()
            if trad is None or self.arret.is_set():
                return
            try:
                fmt, data = _riff(engines.speak(cfg.engine("tts"), trad, voix, lang))
            except Exception as exc:                   # noqa: BLE001
                self.state.report("translation: kokoro failed: %s" % exc)
                continue
            root = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "geshtu-traduction"
            f = root / ("voix-%d.wav" % (n % 2))
            n += 1
            try:
                f.write_bytes(_wav(fmt, data))
            except OSError:
                return
            subprocess.run(["pw-play", str(f)], check=False)
