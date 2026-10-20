"""Live transcript of a recording, in blocks per speaker.

ezvk, 2026-10-20: « je souhaite voir la transcription en temps réel dans la
fenêtre », « on tranche par locuteur », « c'est pour faire des blocs de
compréhension : un locuteur aura un cheminement logique dans sa phrase /
exposé ».

TWO CLOCKS. The capture is read while it grows and cut on PAUSES exactly like
the live translation (traduction.ecoute()); each phrase goes to Whisper and is
on screen about a second after it ends. WHO said it comes from the real
diarisation (diarisation.analyse(), the one the final note uses), run again
every minute or so on everything recorded so far, and matched to the phrases
by time. Consecutive phrases of one speaker form one block; phrases not yet
covered by a diarisation pass wait at the end, unattributed.

⚠️ NOT ONE VOICE PRINT PER PHRASE, AND IT WAS MEASURED. That was the first
version. On the 2026-10-20 meeting (video call, three people), a print on a
single phrase matched the final note's speaker barely half the time:
same-voice and other-voice distances overlap from 0.3 to 0.75. (That note had
itself found four speakers for three people -- `speakers = 4` in the config --
so the figure is rough, but no threshold separated the two.)
Blocks built that way were wrong one time in two and cut a line of thought in
the wrong place -- the opposite of the request. The whole-recording pass
clusters with every phrase at once and is cheap on the NPU (~27 s per hour of
audio), so it is the one that decides.

⚠️ AND IT NEEDS THE NUMBER OF SPEAKERS. Without it the threshold alone found 11
on that meeting, with it 3 -- the window asks for it (Session.locuteurs).

Speaker numbers may move between passes (each pass clusters from scratch over
more audio); the final note is the reference.
"""
from __future__ import annotations

import os
import pathlib
import re
import threading
import time
import wave

from geshtu import engines
from geshtu.traduction import FANTOMES, RATE, SEUIL_VOIX, ecoute

PREMIER = 45.0             # seconds of audio before the first diarisation pass
INTERVALLE = 60.0          # seconds between passes, at least
MAX_BLOCS = 80             # what status() hands the window
# Whisper on silence or music: « Oooooo… », « !!!!!!… ». Not a phrase.
REPETITION = re.compile(r"(.)\1{12,}")


class Direct:
    """One live transcript at a time, tied to the recording that runs."""

    def __init__(self, state) -> None:
        self.state = state
        self.lock = threading.Lock()
        self.arret = threading.Event()
        self.etat = "idle"
        self.phrases: list[dict] = []      # {debut, fin, texte, qui}
        self.session = None

    def status(self) -> dict:
        with self.lock:
            return {"etat": self.etat, "blocs": self._blocs()[-MAX_BLOCS:]}

    def start(self, session, wav: pathlib.Path) -> None:
        """Called by the daemon right after the recorder started -- it must not
        block nor report() synchronously (the daemon holds its lock)."""
        with self.lock:
            self.arret.set()               # a previous transcript, if any, ends
            self.arret = threading.Event()
            self.etat = "running"
            self.phrases, self.session = [], session
        arret = self.arret
        threading.Thread(target=self._ecoute, args=(session, wav, arret), daemon=True).start()
        threading.Thread(target=self._diarise, args=(session, wav, arret), daemon=True).start()

    def stop(self) -> None:
        """Stops listening; the blocks stay on screen until the next start."""
        with self.lock:
            self.arret.set()
            self.etat = "idle"

    # ------------------------------------------------------------------
    def _blocs(self) -> list[dict]:
        """Consecutive phrases of one speaker, one block. Caller holds the lock."""
        nom = self.session.nom if self.session is not None else (lambda q: q)
        blocs: list[dict] = []
        for ph in self.phrases:
            qui = nom(ph["qui"]) if ph["qui"] else "…"
            if blocs and blocs[-1]["qui"] == qui:
                blocs[-1]["texte"] += " " + ph["texte"]
                blocs[-1]["fin"] = ph["fin"]
            else:
                blocs.append({"qui": qui, "debut": ph["debut"], "fin": ph["fin"],
                              "texte": ph["texte"]})
        for b in blocs:
            b["debut"] = _hms(b["debut"])
            b["fin"] = _hms(b["fin"])
        return blocs

    def _ecoute(self, session, wav, arret) -> None:
        clip = session.root / ".direct-phrase.wav"
        ecoute(wav, arret, lambda: SEUIL_VOIX,
               lambda pcm, debut, fin: self._phrase(session, pcm, debut, fin, clip, arret))
        clip.unlink(missing_ok=True)

    def _phrase(self, session, pcm, debut, fin, clip, arret) -> None:
        if arret.is_set():
            return
        with wave.open(str(clip), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        try:
            texte = engines.transcribe(self.state.cfg.engine("asr"), clip,
                                       session.langue_entree).strip()
        except Exception as exc:                       # noqa: BLE001
            self.state.report("live transcript: whisper failed: %s" % exc)
            return
        if len(texte) < 3 or texte.lower() in FANTOMES or REPETITION.search(texte):
            return
        with self.lock:
            if not arret.is_set():
                self.phrases.append({"debut": debut, "fin": fin, "texte": texte, "qui": None})

    # -- who spoke: the real diarisation, again and again -----------------
    def _diarise(self, session, wav, arret) -> None:
        modeles = os.environ.get("GESHTU_DIARISATION")
        if not modeles or not os.path.exists(os.path.join(modeles, "segmentation.xml")):
            self.state.report("live transcript: no diarisation models, no speakers live")
            return
        try:
            from geshtu import diarisation
        except ImportError as exc:
            self.state.report("live transcript: diarisation unavailable: %s" % exc)
            return
        opts = self.state.cfg.raw.get("diarisation", {})
        instantane = session.root / ".direct-diar.wav"
        prochain = PREMIER
        while not arret.wait(5):
            duree = _duree(wav)
            if duree < prochain:
                continue
            t0 = time.time()
            try:
                _instantane(wav, instantane)
                tours = diarisation.analyse(
                    str(instantane), modeles, seuil=opts.get("threshold", 0.9),
                    device=opts.get("device", "NPU"),
                    locuteurs=session.locuteurs or opts.get("speakers"),
                    trace=lambda *_: None)
            except Exception as exc:                   # noqa: BLE001
                self.state.report("live transcript: diarisation pass failed: %s" % exc)
                prochain = duree + INTERVALLE
                continue
            cout = time.time() - t0
            # The same numbering as the final note (builtin/diarise.py): in the
            # order the speakers first appear.
            noms: dict = {}
            for _, _, q in tours:
                noms.setdefault(q, "Speaker %d" % (len(noms) + 1))
            with self.lock:
                if arret.is_set():
                    break
                for ph in self.phrases:
                    if ph["fin"] <= duree:
                        ph["qui"] = _qui(tours, noms, ph["debut"], ph["fin"]) or ph["qui"]
            # A pass costs ~27 s per hour of audio: never run more than half the time.
            prochain = duree + max(INTERVALLE, 2 * cout)
        instantane.unlink(missing_ok=True)


def _qui(tours, noms, debut, fin):
    """The speaker who covers most of [debut, fin] -- as builtin/asr.py does."""
    best, part = None, 0.0
    for d, f, q in tours:
        recouvre = min(f, fin) - max(d, debut)
        if recouvre > part:
            best, part = noms[q], recouvre
    return best


def _data(wav: pathlib.Path) -> tuple[int, int]:
    """Offset of the PCM data and how many bytes of it exist right now. The
    header of a file being recorded does not hold its final size yet."""
    tete = wav.read_bytes()[:4096]
    i = tete.find(b"data")
    debut = i + 8 if i >= 0 else 44
    n = wav.stat().st_size - debut
    return debut, n - n % 2


def _duree(wav: pathlib.Path) -> float:
    try:
        return _data(wav)[1] / 2 / RATE
    except OSError:
        return 0.0


def _instantane(wav: pathlib.Path, dest: pathlib.Path) -> None:
    """Everything recorded so far, as a proper WAV the diarisation can read."""
    debut, n = _data(wav)
    with wav.open("rb") as f:
        f.seek(debut)
        pcm = f.read(n)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def _hms(t: float) -> str:
    t = int(t)
    return "%d:%02d:%02d" % (t // 3600, t % 3600 // 60, t % 60)
