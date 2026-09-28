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
import json
import os
import pathlib
import queue
import re
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
# Language names in English, for TranslateGemma's training prompt.
ANGLAIS = {"fr": "French", "en": "English", "es": "Spanish", "it": "Italian",
           "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
NOMS = {"fr": "français", "en": "anglais", "es": "espagnol", "it": "italien",
        "pt": "portugais", "ru": "russe", "zh": "chinois", "ja": "japonais"}

RATE = 16000
TRAME = 480                 # 30 ms at 16 kHz
SEUIL_VOIX = 700            # int16 RMS, ~ -33 dBFS: below is a pause
PAUSE = 0.6                 # seconds of pause that end a phrase
MIN_VOIX = 0.8              # a phrase needs at least this much speech
MAX_PHRASE = 12.0           # cut anyway: nobody waits longer than this
CONTEXTE = 2                # previous phrases sent with each one, with their translation

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
        self.seuil = SEUIL_VOIX
        self.baisse: tuple[int, float] | None = None   # (node id, volume d'origine)
        self.contexte: list[tuple[str, str]] = []      # (phrase, traduction), les dernières
        self._reprise()

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
            self.seuil = SEUIL_VOIX
            self.baisse = None
            if self.voix and target.kind == "app":
                self._baisser(int(target.key.split(":", 1)[1]))
            self.session, self.src = session, src
            self.etat, self.source, self.cible = "running", entree, cible
            self.lignes = []
            self.contexte = []
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
        self._remonter()
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
                if rms >= self.seuil:
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
        moteurs = cfg.raw.get("traduction", {}).get("moteurs", {})
        choix = moteurs.get(self.state.traducteur_choisi())
        if choix:
            eng = dataclasses.replace(eng, endpoint=choix["endpoint"], model=choix["model"])
            if choix.get("format") == "translategemma":
                return self._translategemma(eng, texte, entree, cible)
        else:
            # Before named translators: a bare [traduction] model on the llm engine.
            modele = cfg.raw.get("traduction", {}).get("model")
            if modele:
                eng = dataclasses.replace(eng, model=modele)
        consigne = ("Tu es un interprète. Traduis fidèlement ce texte parlé du %s "
                    "vers le %s. Réponds UNIQUEMENT par la traduction, sans guillemets, "
                    "sans commentaire, sans note." % (NOMS.get(entree, entree), NOMS[cible]))
        # ⚠️ CONTEXT AS PAST TURNS, not pasted into the prompt. ezvk, 2026-09-28:
        # « donne du contexte ». A phrase cut on a pause loses its subject; the
        # two previous phrases and THEIR translations, given as exchanges already
        # made, carry names, topic and register -- and, being answers the model
        # "already gave", they are not translated a second time.
        trad = engines.chat(eng, texte, system=consigne, max_tokens=300,
                            allow_truncated=True, history=self.contexte[-CONTEXTE:]).strip()
        if trad:
            self.contexte = (self.contexte + [(texte, trad)])[-CONTEXTE:]
        return trad

    # ⚠️ TRANSLATEGEMMA IS PROMPTED WITH ITS OWN TRAINING TURN, verbatim from
    # its model card, sent as the user message through a passthrough chat
    # template (horde hosts/utu/traduction.nix explains why the official
    # template cannot be used). No system prompt, no history: the format has
    # neither, so the context of the previous phrases is not used here.
    # Measured 2026-09-28 on utu's GPU: 1.03 s/phrase, the cleanest French of
    # the local models (no typo, idiomatic), close to Gemma 31B on ishtar.
    def _translategemma(self, eng, texte: str, entree: str, cible: str) -> str:
        src, dst = ANGLAIS.get(entree, entree), ANGLAIS.get(cible, cible)
        invite = ("<start_of_turn>user\nYou are a professional %s (%s) to %s (%s) translator. "
                  "Your goal is to accurately convey the meaning and nuances of the original %s "
                  "text while adhering to %s grammar, vocabulary, and cultural sensitivities.\n"
                  "Produce only the %s translation, without any additional explanations or "
                  "commentary. Please translate the following %s text into %s:\n\n\n"
                  "%s<end_of_turn>\n<start_of_turn>model\n"
                  % (src, entree, dst, cible, src, dst, dst, src, dst, texte))
        # temperature 0: the most likely translation, as measured -- not a draw.
        return engines.chat(eng, invite, max_tokens=300, allow_truncated=True,
                            temperature=0).strip()

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

    # -- lower the source while Kokoro speaks over it ------------------------
    #
    # ezvk, 2026-09-28 : « quand on traduit en presque temps réel il faudrait
    # baisser l'audio de la source pour mieux entendre koko ».
    #
    # ⚠️ THE CAPTURE HEARS THE LOWERED STREAM TOO. geshtu taps the application's
    # output ports, which PipeWire feeds AFTER the stream volume. Lowering the
    # source therefore lowers what Whisper gets and what the pause detector
    # measures -- so the voice threshold is scaled by the same amplitude factor.
    #
    # ⚠️ AND wpctl IS CUBIC. Read on utu (2026-09-28), nothing changed:
    #   speaker   wpctl 0.90 -> channelVolumes 0.728985 (= 0.9^3)
    #   HomePods  wpctl 0.35 -> 0.042875, wpctl 0.51 -> 0.132651
    # An amplitude factor r is a wpctl factor r^(1/3).
    # ⚠️ A DAEMON KILLED MID-TRANSLATION LEFT THE SOURCE LOWERED FOR GOOD. Seen
    # on utu 2026-09-28: a deploy restarted geshtu while Firefox was at 0.50;
    # the next translation read 0.50 as the "original" and went to 0.25, and
    # WirePlumber stored it -- every Firefox stream then started at 1.6 %
    # amplitude. ezvk: « il est cassé le son ». The original volume is now
    # written to disk BEFORE lowering, and a new daemon puts it back.
    def _fichier_baisse(self) -> pathlib.Path:
        return (pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
                / "geshtu-traduction-baisse.json")

    def _reprise(self) -> None:
        f = self._fichier_baisse()
        try:
            node, v0 = json.loads(f.read_text())
        except (OSError, ValueError):
            return
        try:
            r = subprocess.run(["wpctl", "set-volume", str(node), "%.3f" % v0],
                               capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            self.state.report("translation: could not restore node %s: %s" % (node, exc))
            return
        self.state.report("translation: a previous daemon left node %s lowered -- "
                          "restored to %.2f%s" % (node, v0, "" if r.returncode == 0
                                                  else " (stream gone: raise it by hand)"))
        f.unlink(missing_ok=True)

    def _baisser(self, node: int) -> None:
        db = float(self.state.cfg.raw.get("traduction", {}).get("attenuation_db", 12))
        if db <= 0:
            return
        try:
            out = subprocess.run(["wpctl", "get-volume", str(node)],
                                 capture_output=True, text=True, timeout=5).stdout
            v0 = float(re.search(r"Volume:\s*([\d.]+)", out).group(1))
        except Exception as exc:                       # noqa: BLE001
            self.state.report("translation: could not read the source volume: %s" % exc)
            return
        r = 10 ** (-db / 20)
        self._fichier_baisse().write_text(json.dumps([node, v0]))
        subprocess.run(["wpctl", "set-volume", str(node), "%.3f" % (v0 * r ** (1 / 3))],
                       capture_output=True, timeout=5)
        self.baisse = (node, v0)
        self.seuil = SEUIL_VOIX * r
        self.state.report("translation: source lowered by %.0f dB (wpctl %.2f -> %.2f)"
                          % (db, v0, v0 * r ** (1 / 3)))

    def _remonter(self) -> None:
        if not self.baisse:
            return
        node, v0 = self.baisse
        self.baisse = None
        # The stream may be gone (tab closed): nothing to restore, and no harm.
        try:
            subprocess.run(["wpctl", "set-volume", str(node), "%.3f" % v0],
                           capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            self.state.report("translation: could not restore the source: %s" % exc)
        self._fichier_baisse().unlink(missing_ok=True)
