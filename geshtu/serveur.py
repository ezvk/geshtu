"""geshtu's own model servers: OpenVINO Model Server processes started when a
stage needs a model, and stopped when nothing has used them for a while.

ezvk, 2026-11-28: « on va integrer le serveur openvino directement dans geshtu
comme ca on charge que ce qui est necessaire quand c est necessaire et ovms
gui pour le reste ». Before, every model was a system service listening all
day, whether a meeting was on or not.

An engine is MANAGED when it names a server and a model directory:

    [serveurs.stable]
    ovms = "/run/current-system/sw/bin/ovms"
    port = 18194

    [engines.asr]
    serveur = "stable"
    modele = "/var/lib/ovmodels/OpenVINO/whisper-large-v3-turbo-int8-ov"
    model = "whisper"          # the name served, as before
    device = "NPU"

Its endpoint is then derived (127.0.0.1, the server's port, the route of the
task). Three levels of "only when needed":

  - the PROCESS starts at the first request that needs it, and stops after
    `arret` seconds without one;
  - a MODEL enters the server's config the first time it is asked for;
  - its WEIGHTS load at the request and unload after `inactif` seconds
    without one (--idle_unload_timeout_seconds, OVMS's own mechanism).

⚠️ ONE SERVER PER OVMS VERSION, NOT PER DEVICE. A process serves every
device; what differs between versions is speed and support. Measured on utu
(2026-11-23): Whisper on the NPU takes 0.37-0.40 s per clip on the stable
2026.4.1 and 2.10-2.18 s on the weekly 2026.5.0.dbe11fa27, while LFM3-VL is
only served by that weekly. Hence two servers there, not one.

⚠️ THE MODEL DIRECTORY IS NEVER WRITTEN TO. `ovms --configure` writes its
graph.pbtxt next to the weights (see repository.py), and the shared store
belongs to root. Each managed model gets a directory of ours instead: a link
to every file of the source, plus its own graph.pbtxt and any file the
engine replaces (`fichiers`, e.g. a chat template). It is rebuilt when the
source, the task, the device, the options or the OVMS binary change.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import threading
import time
import urllib.request

ROUTES = {
    "text_generation": "/v3/chat/completions",
    "embeddings": "/v3/embeddings",
    "rerank": "/v3/rerank",
    "speech2text": "/v3/audio/transcriptions",
    "text2speech": "/v3/audio/speech",
    "image_generation": "/v3/images/generations",
}

# The task an engine serves when it does not say: what the stage using it calls.
TACHES = {"asr": "speech2text", "tts": "text2speech", "embed": "embeddings"}

GABARITS = pathlib.Path(__file__).parent / "gabarits"


def endpoint(port: int, tache: str) -> str:
    return "http://127.0.0.1:%d%s" % (int(port), ROUTES[tache])


def fichier_source(valeur: str) -> pathlib.Path:
    """`geshtu:nom` = a file shipped with geshtu (geshtu/gabarits), else a path."""
    if valeur.startswith("geshtu:"):
        return GABARITS / valeur[len("geshtu:"):]
    return pathlib.Path(valeur).expanduser()


def _graph_device(path: pathlib.Path) -> str:
    try:
        text = (path / "graph.pbtxt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    import re
    m = re.search(r'(?:target_)?device:\s*"([^"]*)"', text)
    return m.group(1) if m else ""


class Serveur:
    """One OVMS process, its config and the model directories it serves."""

    def __init__(self, nom: str, ovms: str, port: int, racine: pathlib.Path,
                 inactif: int = 180, arret: int = 900, cache: bool = False,
                 options: str = "", report=print):
        self.nom = nom
        self.ovms = ovms
        self.port = int(port)
        self.dossier = racine / nom
        self.inactif = int(inactif)
        self.arret = int(arret)
        self.cache = bool(cache)
        self.options = options
        self.report = report
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.servis: set[str] = set()
        self.moteurs: list = []          # every engine of the config served here
        self.dernier = 0.0
        self.en_cours = 0

    # -- the model directories ------------------------------------------
    @property
    def config(self) -> pathlib.Path:
        return self.dossier / "config.json"

    def servable(self, eng) -> pathlib.Path:
        """The directory OVMS serves for this engine, (re)built if needed."""
        src = pathlib.Path(eng.modele).expanduser()
        dest = self.dossier / "modeles" / eng.model
        fichiers = dict(eng.fichiers or {})
        empreinte = hashlib.sha256(json.dumps(
            [str(src), eng.tache, eng.device, eng.options, sorted(fichiers.items()),
             self.ovms]).encode()).hexdigest()
        marque = dest / ".geshtu"
        try:
            if marque.read_text() == empreinte:
                return dest
        except OSError:
            pass
        if not any(src.glob("openvino_*.xml")):
            raise RuntimeError("%s: no OpenVINO model (openvino_*.xml) in %s" % (eng.name, src))
        remplaces = {nom: fichier_source(v) for nom, v in fichiers.items()}
        for nom, f in remplaces.items():
            if not f.is_file():
                raise RuntimeError("%s: %s replaces %s but does not exist" % (eng.name, f, nom))
        if dest.exists():
            shutil.rmtree(dest)         # links and our own files only: the source stays
        dest.mkdir(parents=True)
        for entree in sorted(src.iterdir()):
            if entree.name != "graph.pbtxt" and entree.name not in remplaces:
                os.symlink(entree, dest / entree.name)
        for nom, f in remplaces.items():
            shutil.copyfile(f, dest / nom)
        graphe = src / "graph.pbtxt"
        if graphe.is_file() and not eng.options and eng.device in ("", _graph_device(src)):
            # The model's own graph, kept as is: it may carry settings of its own.
            shutil.copyfile(graphe, dest / "graph.pbtxt")
        else:
            args = [self.ovms, "--configure", "--model_path", str(dest), "--task", eng.tache]
            if eng.device:
                args += ["--target_device", eng.device]
            r = subprocess.run(args + shlex.split(eng.options or ""), capture_output=True,
                               text=True, timeout=120)
            if r.returncode != 0 or not (dest / "graph.pbtxt").is_file():
                shutil.rmtree(dest, ignore_errors=True)
                raise RuntimeError("%s: ovms --configure failed: %s"
                                   % (eng.name, (r.stdout + r.stderr).strip()[-400:]))
        marque.write_text(empreinte)
        return dest

    def _ecrit_config(self) -> None:
        cfg = {"model_config_list": [
            {"config": {"name": n, "base_path": str(self.dossier / "modeles" / n)}}
            for n in sorted(self.servis)]}
        tmp = self.config.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2))
        tmp.replace(self.config)

    # -- the process -------------------------------------------------------
    def vivant(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _demarre(self) -> None:
        """Start with EVERY model of this server in the config, not only the one asked.

        ⚠️ A CONFIG CHANGED UNDER A RUNNING SERVER RELOADS ITS OTHER GRAPHS.
        Measured on utu (2026-11-28, stable 2026.4.1): adding gemma4 to a server
        that held kokoro made it « Reloading mediapipe graph: kokoro » -- an NPU
        recompile of ~30 s with no cache -- and the request sent meanwhile got a
        bare 500. Declared all at start, the four models stayed SLEEPING until
        asked and nothing was ever reloaded: declaring a model loads nothing.
        """
        self.dossier.mkdir(parents=True, exist_ok=True)
        for autre in self.moteurs:
            try:
                self.servable(autre)
                self.servis.add(autre.model)
            except Exception as exc:                     # noqa: BLE001
                self.report("server %s: %s left out: %s" % (self.nom, autre.model, exc))
        self._ecrit_config()
        args = [self.ovms, "--config_path", str(self.config),
                "--rest_bind_address", "127.0.0.1", "--rest_port", str(self.port),
                "--file_system_poll_wait_seconds", "1",
                "--log_level", "INFO"]
        if self.inactif > 0:            # 0 = weights stay loaded while the process runs
            args += ["--idle_unload_timeout_seconds", str(self.inactif)]
        if self.cache:
            (self.dossier / "cache").mkdir(exist_ok=True)
            args += ["--cache_dir", str(self.dossier / "cache")]
        args += shlex.split(self.options)
        journal = open(self.dossier / "ovms.log", "w")
        self.proc = subprocess.Popen(args, cwd=self.dossier, stdout=journal,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        journal.close()
        self.report("server %s: started (pid %d, :%d)" % (self.nom, self.proc.pid, self.port))

    def arrete(self, pourquoi: str = "") -> bool:
        with self.lock:
            if not self.vivant():
                return False
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
            self.report("server %s: stopped%s" % (self.nom, " (%s)" % pourquoi if pourquoi else ""))
            return True

    def journal(self, n: int = 6) -> str:
        try:
            lignes = (self.dossier / "ovms.log").read_text(errors="replace").splitlines()
        except OSError:
            return ""
        erreurs = [line for line in lignes if "][error]" in line or "Exception" in line]
        return "\n".join((erreurs or lignes)[-n:])

    def etats(self) -> dict:
        """{model: state} as the running server reports it ({} when stopped)."""
        if not self.vivant():
            return {}
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/config" % self.port,
                                        timeout=5) as r:
                raw = json.loads(r.read())
        except Exception:                                # noqa: BLE001
            return {}
        out = {}
        for nom, info in raw.items():
            versions = info.get("model_version_status") or []
            out[nom] = versions[0].get("state") if versions else "?"
        return out

    # -- what a request calls ---------------------------------------------
    def assure(self, eng, attente: float = 180) -> None:
        """The server runs and serves this engine's model; returns once it does."""
        with self.lock:
            self.servable(eng)
            if eng.model not in self.servis:
                # Not known at start (left out, or not in the config then): a
                # restart, never an edit of a running server's config (_demarre).
                self.servis.add(eng.model)
                if self.vivant():
                    self.arrete("%s added" % eng.model)
            if not self.vivant():
                self._demarre()
            self.dernier = time.time()
        fin = time.time() + attente
        while time.time() < fin:
            if not self.vivant():
                raise RuntimeError("server %s exited (code %s): %s"
                                   % (self.nom, self.proc.returncode, self.journal()))
            if self.etats().get(eng.model) == "AVAILABLE":
                return
            time.sleep(0.3)
        raise RuntimeError("server %s: %s not available after %d s: %s"
                           % (self.nom, eng.model, attente, self.journal()))

    def debut(self) -> None:
        with self.lock:
            self.en_cours += 1
            self.dernier = time.time()

    def fin(self) -> None:
        with self.lock:
            self.en_cours = max(0, self.en_cours - 1)
            self.dernier = time.time()

    def veille(self) -> None:
        """Stop the process once nothing has asked for it for `arret` seconds."""
        with self.lock:
            if (self.vivant() and not self.en_cours and self.arret > 0
                    and time.time() - self.dernier > self.arret):
                self.arrete("unused for %d s" % self.arret)


class Serveurs:
    """Every server of the config, and the thread that stops idle ones."""

    def __init__(self, cfg, report=print):
        racine = cfg.sessions / "serveurs"
        self.tous = {
            nom: Serveur(nom, d["ovms"], d["port"], racine, d.get("inactif", 180),
                         d.get("arret", 900), d.get("cache", False), d.get("options", ""),
                         report)
            for nom, d in cfg.raw.get("serveurs", {}).items()}
        geres = [e for e in cfg.engines.values() if e.serveur]
        geres += [t for t in (cfg.traducteur(n) for n in
                              cfg.raw.get("traduction", {}).get("moteurs", {})) if t and t.serveur]
        for eng in geres:
            if eng.serveur in self.tous:
                self.tous[eng.serveur].moteurs.append(eng)
        threading.Thread(target=self._veille, daemon=True).start()

    def _veille(self) -> None:
        while True:
            time.sleep(15)
            for s in self.tous.values():
                try:
                    s.veille()
                except Exception:                        # noqa: BLE001
                    pass

    def de(self, eng) -> Serveur:
        try:
            return self.tous[eng.serveur]
        except KeyError:
            raise RuntimeError("%s: no [serveurs.%s] declared" % (eng.name, eng.serveur))

    def etat(self) -> list[dict]:
        out = []
        for nom, s in sorted(self.tous.items()):
            out.append({"serveur": nom, "port": s.port, "ovms": s.ovms, "actif": s.vivant(),
                        "modeles": s.etats(), "declares": sorted(s.servis)})
        return out

    def arrete_tout(self, pourquoi: str = "") -> list[str]:
        return [nom for nom, s in self.tous.items() if s.arrete(pourquoi)]


# The daemon's servers. engines.* calls `utilise(engine)` around each request:
# a managed engine with no daemon behind it is an error, not a silent fallback.
GESTION: Serveurs | None = None


class _Utilisation:
    def __init__(self, eng):
        self.eng = eng
        self.s: Serveur | None = None

    def __enter__(self):
        if getattr(self.eng, "serveur", ""):
            if GESTION is None:
                raise RuntimeError("%s is served by geshtud (serveur %r), which is not running "
                                   "in this process" % (self.eng.name, self.eng.serveur))
            self.s = GESTION.de(self.eng)
            self.s.debut()
            try:
                self.s.assure(self.eng)
            except Exception:
                self.s.fin()
                raise
        return self

    def __exit__(self, *exc):
        if self.s:
            self.s.fin()
        return False


def utilise(eng) -> _Utilisation:
    """`with serveur.utilise(engine):` around a request to that engine."""
    return _Utilisation(eng)
