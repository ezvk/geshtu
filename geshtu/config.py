"""Configuration, and the engine table that makes NPU/GPU placement a setting.

The whole point of separating engines from stages is that WHERE a model runs
is a deployment fact, not a code fact. A stage asks for the engine named
"asr"; whether that is an NPU at 40x realtime, an iGPU, or a remote box is
decided here.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import tomllib

DEFAULT = """
[paths]
# sessions = "~/.local/share/geshtu"

[capture]
rate = 16000
channels = 1

[engines.asr]
endpoint = "http://localhost:8094/v3/audio/transcriptions"
model = "whisper"
device = "NPU"
# format = "omni" sends the audio to an omni-modal model (/chat/completions)
# instead. ⚠️ Measured worse in French (2026-11-20, FLEURS + MLS, 980 clips):
# Whisper turbo 5.6 % WER, LFM3-Omni-2B 7.6 % -- Whisper stays the default.

[engines.llm]
endpoint = "http://localhost:8092/v3/chat/completions"
model = "qwen3-coder-30b"
# ⚠️ `device` IS A CLAIM, NOT A CONTROL. It does not move anything: which
# accelerator serves a model is decided when the model server starts. It
# exists so the pipeline can reason about this endpoint's limits -- and it
# can therefore be WRONG, which is worse than absent. Measured: this file
# said CPU for a week after the endpoint had moved to the GPU, and nothing
# anywhere contradicted it. Check it against the server, not against memory.
device = "GPU"
# The prompt ceiling of this endpoint, in tokens. 0 means no known limit.
#
# ⚠️ IT IS A PROPERTY OF THE ACCELERATOR, NOT OF THE MODEL. Measured on an
# Intel NPU (2026-09-19) by bisection: 8192 compiles in 75 s, 9216 is refused
# in 15 s, and a model a gigabyte smaller fails at exactly the same point --
# a compiler constant, not a memory limit. The same weights on the GPU
# accepted 21 043 tokens and answered correctly (2026-09-21).
#
# An hour of speech is 10,000-15,000 tokens, so this number decides whether a
# meeting is summarised whole or in chapters.
max_prompt = 0

[engines.embed]
endpoint = "http://localhost:8096/v3/embeddings"
# The name the SERVER serves, which is rarely the name of the weights.
# `geshtu models` prints what each endpoint actually offers; a name that
# matches nothing fails at the first request, an hour after the recording.
model = "embeddings"
device = "GPU"

# Live translation (geshtu/traduction.py): NAMED translators, one picked in the
# window's "Traducteur" menu and remembered by the daemon. ezvk, 2026-09-28:
# « par sélection dans le menu de gesh, chargé à la demande ».
# `local` = TranslateGemma 4B on utu's GPU, woken on demand behind :8092
# (horde hosts/utu/traduction.nix) -- works off the tailnet. `format =
# "translategemma"`: the model's own training prompt (see traduction.py).
# Add others in ~/.config/geshtu/geshtu.toml, e.g. [traduction.moteurs.ishtar].
[traduction.moteurs.local]
endpoint = "http://localhost:8092/v3/chat/completions"
model = "translategemma"
format = "translategemma"

[engines.tts]
# Read-aloud of the latest summary and of the selected text (geshtu/lecture.py).
# Kokoro on the NPU.
# ⚠️ The language is sent per request (fr-fr / en-us…, from the summary's own
# language, or guessed from the selection's function words, else [summary]
# language): without it Kokoro phonemises French with English rules.
endpoint = "http://localhost:8098/v3/audio/speech"
model = "kokoro"
device = "NPU"

# [lecture]
# voix_fr = "ff_siwis"    # the model's only French voice
# voix_en = "af_heart"

# Who spoke when. Absent models simply skip the stage.
# [diarisation]
# engine = "openvino"   # or "sherpa", the CPU tool kept as a reference
# device = "NPU"        # measured fastest of the three, and silent
# speakers = 4          # when you know: a count always beats a threshold
#
# ⚠️ `threshold` is a cosine distance and it is SHARP. Measured on a 2 h 35
# conference: 0.5 gave 145 speakers, 0.7 gave 43, 0.9 gave 4, 1.1 gave 1.
# Higher merges. The default sits at 0.9 because the same voice recorded
# through a room microphone and through a remote link drifts well past 0.5
# from itself.
# threshold = 0.9
# threads = 4           # sherpa only

# Voice command matching. The table itself (formulations, actions) lives
# in a JSON file, not here -- see geshtu/commandes.py's own precedence:
# ~/.config/geshtu/commandes.json first, then /etc/geshtu/commandes.json.
# [dictation]
#
# The spoken language is NOT set here: see [transcription] below, and the
# daemon's remembered choice (`geshtu entree`), which apply to dictation too.
#
# threshold = 0.75     # overrides the table's own "seuil" if set

# The shared screen (geshtu/ecran.py, stage `ecrans`). Each state of the
# screen that changed and then held still is kept as a picture, read by the
# `vision` engine, and placed in the notes at its moment.
# ⚠️ OFF BY DEFAULT: the live capture sees the WHOLE screen, notifications
# included. `sortie` limits it to one output (grim -o), `zone` to a region
# (grim -g "x,y wxh", e.g. from `slurp`). A video file is always looked at.
[ecran]
actif = false
# sortie = "eDP-1"
# zone = "0,0 1560x1040"
# fenetre = "teams-for-linux"   # ONE window, by app-id or a piece of its title:
#   captured even on a hidden workspace and with the screen asleep (grim -T);
#   the whole-screen capture sees nothing while the screen is off.
# Measured on a 15 min conference (2026-11-20), 0 slide state lost of 29:
# pixel = 25       # grey-level difference that counts as a changed pixel
# cellule = 0.03   # share of ONE grid cell changed since the last picture kept
# mouvement = 0.003
# stable = 2       # seconds the screen must hold still
# vivant = 5       # a cell moving on its own this often per minute is video
#
# The engine that reads the screens. An omni or vision model behind
# /chat/completions (OpenAI image_url). Absent: the pictures are kept, unread.
# [engines.vision]
# endpoint = "http://localhost:8101/v3/chat/completions"
# model = "lfm3-omni"
# device = "NPU"

[transcription]
# The SPOKEN language sent to Whisper, when nothing else was chosen. The
# window's "Audio en" menu (or `geshtu entree`) is remembered and wins.
# ⚠️ Whisper on OVMS does not detect per request (engines.transcribe()).
language = "fr"

[summary]
# The summary language when nothing else was chosen. The choice made in the
# window (or `geshtu lang`) is remembered by the daemon and wins over this.
language = "fr"

[pipeline]
# ⚠️ `diarise` AVANT `transcribe`, ET CE N EST PAS INDIFFERENT. Les tranches
# de transcription sont coupees sur les SILENCES, les tours de parole sur les
# CHANGEMENTS DE VOIX. Transcrire d abord fige une grille de deux minutes dans
# laquelle une interjection de trois secondes ne peut pas exister : mesure du
# 2026-09-21, trois locuteurs trouves et une seule etiquette dans la note.
# ⚠️ PLUS DE RÉSUMÉ PAR DÉFAUT. ezvk, 2026-10-20 : « on va retirer le summarize,
# c'est juste une transcription d'un flux audio » ; puis « garde la diarisation ».
# Chapitrage et résumé restent installés :
# les remettre ici (ou dans ~/.config/geshtu/geshtu.toml) les réactive --
#   stages = ["diarise", "transcribe", "chapter", "summarise"]
stages = ["diarise", "transcribe", "ecrans"]
# `mp3` AVANT `markdown` : l'enregistrement gardé en MP3 à côté de la note, puis
# les WAV effacés (ezvk, 2026-10-21 : « 1 en mp3 fais le menage »). La note
# ne lie l'audio que s'il existe déjà.
sinks = ["mp3", "markdown"]

# geshtu's own model servers (geshtu/serveur.py). An engine with `serveur` and
# `modele` is served by geshtu itself: the OVMS process starts at the first
# request that needs it and stops after `arret` seconds without one; a model's
# weights unload after `inactif` seconds (OVMS --idle_unload_timeout_seconds).
# One server per OVMS VERSION (it serves every device): utu runs Whisper on the
# stable (5x faster on the NPU) and LFM3-VL on a weekly (the only one serving it).
# [serveurs.stable]
# ovms = "/run/current-system/sw/bin/ovms"
# port = 18194
# inactif = 600          # seconds before a model's weights unload
# arret = 900            # seconds before the process itself stops
# cache = true           # --cache_dir: the NPU recompiles Whisper otherwise (~3 min)
#
# [engines.asr]
# serveur = "stable"
# modele = "/var/lib/ovmodels/OpenVINO/whisper-large-v3-turbo-int8-ov"
# model = "whisper"
# device = "NPU"
# options = ""           # ovms --configure options, e.g. "--reasoning_parser qwen3"
# fichiers = { "chat_template.jinja" = "geshtu:translategemma.jinja" }
#                        # files replaced in geshtu's copy; geshtu: = shipped with geshtu

# Dropping a model in a folder and having it served. The server must be
# started with --config_path <config> --file_system_poll_wait_seconds N,
# otherwise it never re-reads what geshtu writes here.
# [repository]
# ovms = "ovms"
# models = "/var/lib/geshtu/models"
# config = "/var/lib/geshtu/ovms.json"
"""


@dataclasses.dataclass
class Engine:
    name: str
    endpoint: str = ""
    model: str = ""
    device: str = "CPU"
    max_prompt: int | None = None
    timeout: float = 600.0
    # How this endpoint is spoken to, when it is not the obvious way. Only the
    # asr engine reads it: "" = Whisper's multipart /audio/transcriptions,
    # "omni" = an omni-modal model behind /chat/completions (engines.transcribe).
    format: str = ""
    # A MANAGED engine (geshtu/serveur.py): geshtu starts the [serveurs.<serveur>]
    # OVMS itself and serves `modele` (a model directory, never written to) under
    # the name `model`. Its endpoint is derived from the server's port and the task.
    serveur: str = ""
    modele: str = ""
    tache: str = ""                 # empty = the task this engine's stage calls
    options: str = ""               # passed to ovms --configure, e.g. "--pooling LAST"
    fichiers: dict = dataclasses.field(default_factory=dict)   # name -> replacement file


def engine(name: str, raw: dict, fields: dict) -> Engine:
    """An Engine from its table, with a managed one's endpoint filled in."""
    from geshtu import serveur
    eng = Engine(name=name, **fields)
    if eng.serveur:
        port = raw.get("serveurs", {}).get(eng.serveur, {}).get("port")
        if port is None:
            raise SystemExit("[engines.%s] serveur = %r, but no [serveurs.%s] with a port"
                             % (name, eng.serveur, eng.serveur))
        tache = eng.tache or serveur.TACHES.get(name, "text_generation")
        # ⚠️ ALWAYS DERIVED, never the table's: the built-in defaults give every
        # engine an endpoint, and a merged one would point at the old service.
        eng = dataclasses.replace(eng, tache=tache, endpoint=serveur.endpoint(port, tache))
    return eng


@dataclasses.dataclass
class Config:
    sessions: pathlib.Path
    rate: int
    channels: int
    engines: dict[str, Engine]
    stages: list[str]
    sinks: list[str]
    raw: dict

    def engine(self, name: str) -> Engine:
        try:
            return self.engines[name]
        except KeyError:
            raise SystemExit(
                "no engine named %r; declare [engines.%s] in the config" % (name, name))

    def traducteur(self, nom: str) -> Engine | None:
        """[traduction.moteurs.<nom>] as an engine: the llm engine's settings, with
        the translator's own endpoint and model -- or its own managed model."""
        choix = self.raw.get("traduction", {}).get("moteurs", {}).get(nom)
        if not choix:
            return None
        base = self.engines.get("llm") or Engine(name="llm")
        champs = {f.name: getattr(base, f.name) for f in dataclasses.fields(Engine)
                  if f.name not in ("name", "endpoint", "model", "serveur", "modele", "tache",
                                    "options", "fichiers", "format")}
        champs.update(choix)
        return engine("traduction." + nom, self.raw, champs)

    def with_models(self, chosen: dict, endpoints: dict | None = None) -> "Config":
        """A copy of this config with some engines pointed at another model.

        ⚠️ WHICH MODEL IS RUNNING CHANGES UNDER YOU. A model server is
        reconfigured, a model is swapped for a smaller one to free the
        accelerator, a name gains a version suffix. Pinning the name in a file
        that only a rebuild can change turns an ordinary Tuesday into an
        outage, so the choice is runtime state, kept beside the sessions.
        """
        endpoints = endpoints or {}
        if not chosen and not endpoints:
            return self
        engines = {}
        for name, eng in self.engines.items():
            # ⚠️ NOT FOR A MANAGED ENGINE: its name is the one geshtu serves its
            # `modele` under, and another name would be asked of a server that
            # has no such model.
            if chosen.get(name) and not eng.serveur:
                eng = dataclasses.replace(eng, model=chosen[name])
            # Same reasoning for WHERE the model runs: a server started by
            # hand (Jan's llama, a test box) is runtime state, not config.
            # Pointing a managed engine elsewhere takes it out of geshtu's hands.
            if endpoints.get(name):
                eng = dataclasses.replace(eng, endpoint=endpoints[name], serveur="")
            engines[name] = eng
        return dataclasses.replace(self, engines=engines)


def path() -> pathlib.Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (pathlib.Path.home() / ".config")
    return pathlib.Path(base) / "geshtu" / "geshtu.toml"


def load(explicit: pathlib.Path | None = None) -> Config:
    p = explicit or path()
    raw = tomllib.loads(DEFAULT)
    if p.is_file():
        merge(raw, tomllib.loads(p.read_text()))
    data = pathlib.Path(
        os.environ.get("XDG_DATA_HOME") or (pathlib.Path.home() / ".local/share"))
    sessions = raw.get("paths", {}).get("sessions")
    return Config(
        sessions=pathlib.Path(sessions).expanduser() if sessions else data / "geshtu",
        rate=raw["capture"]["rate"],
        channels=raw["capture"]["channels"],
        engines={k: engine(k, raw, v) for k, v in raw.get("engines", {}).items()},
        stages=raw["pipeline"]["stages"],
        sinks=raw["pipeline"]["sinks"],
        raw=raw,
    )


def merge(base: dict, over: dict) -> None:
    """Deep merge, so a user config may set one key of one engine without
    having to restate the whole table."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
