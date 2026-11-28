"""geshtu's own model servers (geshtu/serveur.py), against tests/faux_ovms.py."""
import json
import os
import pathlib
import socket
import sys
import time

import pytest

from geshtu import config, engines, serveur

FAUX = pathlib.Path(__file__).parent / "faux_ovms.py"


def port_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def monde(tmp_path, monkeypatch):
    """A model store we must not touch, a config, and the daemon's servers."""
    os.chmod(FAUX, 0o755)
    lanceur = tmp_path / "ovms"
    lanceur.write_text("#!/bin/sh\nexec %s %s \"$@\"\n" % (sys.executable, FAUX))
    lanceur.chmod(0o755)
    store = tmp_path / "store"
    for nom in ("llm", "trad"):
        d = store / nom
        d.mkdir(parents=True)
        (d / "openvino_model.xml").write_text("<net/>")
        (d / "chat_template.jinja").write_text("officiel")
    (store / "asr").mkdir()
    (store / "asr" / "openvino_encoder_model.xml").write_text("<net/>")
    (store / "asr" / "graph.pbtxt").write_text('device: "NPU"\n')
    toml = tmp_path / "geshtu.toml"
    toml.write_text("""
[paths]
sessions = "%(t)s/sessions"
[serveurs.essai]
ovms = "%(o)s"
port = %(p)d
arret = 1
[engines.llm]
serveur = "essai"
modele = "%(s)s/llm"
model = "gemma"
device = "GPU"
options = "--reasoning_parser gemma4"
[engines.asr]
serveur = "essai"
modele = "%(s)s/asr"
model = "whisper"
[engines.embed]
endpoint = "http://localhost:9/v3/embeddings"
model = "ailleurs"
[traduction.moteurs.local]
serveur = "essai"
modele = "%(s)s/trad"
model = "translategemma"
format = "translategemma"
fichiers = { "chat_template.jinja" = "geshtu:translategemma.jinja" }
""" % {"t": tmp_path, "o": lanceur, "p": port_libre(), "s": store})
    cfg = config.load(toml)
    avant = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    serveur.GESTION = serveur.Serveurs(cfg, report=lambda _l: None)
    yield cfg, store
    serveur.GESTION.arrete_tout()
    serveur.GESTION = None
    # ⚠️ THE STORE IS NEVER WRITTEN TO, whatever the test did.
    assert {p: p.read_bytes() for p in store.rglob("*") if p.is_file()} == avant


def test_endpoint_derived_from_server_and_task(monde):
    cfg, _ = monde
    port = cfg.raw["serveurs"]["essai"]["port"]
    assert cfg.engine("llm").endpoint == "http://127.0.0.1:%d/v3/chat/completions" % port
    assert cfg.engine("asr").tache == "speech2text"
    assert cfg.engine("asr").endpoint.endswith("/v3/audio/transcriptions")
    embed = cfg.engine("embed")
    assert embed.serveur == "" and embed.endpoint.endswith(":9/v3/embeddings")


def test_nothing_runs_until_asked(monde):
    cfg, _ = monde
    s = serveur.GESTION.tous["essai"]
    assert not s.vivant()
    # Listing what an engine serves must not start anything.
    assert engines.available(cfg.engine("llm")) == [
        {"name": "gemma", "state": "on demand (geshtu, essai)"}]
    assert not s.vivant()


def test_first_request_starts_and_serves(monde):
    cfg, store = monde
    eng = cfg.engine("llm")
    assert engines.chat(eng, "bonjour") == "gemma: bonjour"
    s = serveur.GESTION.tous["essai"]
    assert s.vivant()
    d = s.dossier / "modeles" / "gemma"
    assert (d / "openvino_model.xml").is_symlink()
    assert (d / "openvino_model.xml").resolve() == store / "llm" / "openvino_model.xml"
    graphe = (d / "graph.pbtxt").read_text()
    assert 'task: "text_generation"' in graphe and 'device: "GPU"' in graphe
    assert "--reasoning_parser gemma4" in graphe
    argv = json.loads((s.dossier / "argv.json").read_text())
    assert "--idle_unload_timeout_seconds" in argv and "--file_system_poll_wait_seconds" in argv
    assert engines.available(eng) == [{"name": "gemma", "state": "AVAILABLE"}]


def test_every_model_declared_at_start(monde):
    cfg, _ = monde
    engines.chat(cfg.engine("llm"), "un")
    s = serveur.GESTION.tous["essai"]
    # The translator and the asr engine are in the config from the start: asking
    # for them later never edits the config of the running server.
    assert s.servis == {"gemma", "whisper", "translategemma"}
    config_avant = s.config.read_text()
    pid = s.proc.pid
    trad = cfg.traducteur("local")
    assert trad.format == "translategemma" and trad.serveur == "essai"
    assert engines.chat(trad, "deux") == "translategemma: deux"
    assert s.proc.pid == pid and s.config.read_text() == config_avant
    copie = s.dossier / "modeles" / "translategemma" / "chat_template.jinja"
    assert not copie.is_symlink()
    assert copie.read_text() == (serveur.GABARITS / "translategemma.jinja").read_text()


def test_model_graph_kept_when_nothing_overrides_it(monde):
    cfg, _ = monde
    s = serveur.GESTION.tous["essai"]
    d = s.servable(cfg.engine("asr"))
    assert (d / "graph.pbtxt").read_text() == 'device: "NPU"\n'
    assert not (d / "graph.pbtxt").is_symlink()


def test_changed_options_rebuild(monde):
    cfg, _ = monde
    import dataclasses
    s = serveur.GESTION.tous["essai"]
    eng = cfg.engine("llm")
    s.servable(eng)
    s.servable(dataclasses.replace(eng, options="--tool_parser gemma4"))
    assert "--tool_parser gemma4" in (s.dossier / "modeles" / "gemma" / "graph.pbtxt").read_text()


def test_configure_failure_is_reported_and_leaves_nothing(monde):
    cfg, _ = monde
    import dataclasses
    s = serveur.GESTION.tous["essai"]
    with pytest.raises(RuntimeError, match="--refuse"):
        s.servable(dataclasses.replace(cfg.engine("llm"), options="--refuse"))
    assert not (s.dossier / "modeles" / "gemma").exists()


def test_idle_server_stops_and_comes_back(monde):
    cfg, _ = monde
    s = serveur.GESTION.tous["essai"]
    engines.chat(cfg.engine("llm"), "un")
    s.dernier -= 5                                 # arret = 1 s in this config
    s.veille()
    assert not s.vivant()
    assert engines.chat(cfg.engine("llm"), "deux") == "gemma: deux"
    assert s.vivant()


def test_busy_server_is_not_stopped(monde):
    cfg, _ = monde
    s = serveur.GESTION.tous["essai"]
    with serveur.utilise(cfg.engine("llm")):
        s.dernier -= 5
        s.veille()
        assert s.vivant()


def test_endpoint_override_takes_it_out_of_geshtu(monde):
    cfg, _ = monde
    autre = cfg.with_models({"llm": "autre-nom"}, {"llm": "http://localhost:1/v3/chat/completions"})
    assert autre.engine("llm").serveur == ""
    # A model choice alone does not rename a managed model.
    assert cfg.with_models({"llm": "autre-nom"}).engine("llm").model == "gemma"


def test_managed_engine_without_daemon_is_an_error(monde):
    cfg, _ = monde
    serveur.GESTION, gardee = None, serveur.GESTION
    try:
        with pytest.raises(RuntimeError, match="geshtud"):
            engines.chat(cfg.engine("llm"), "x")
    finally:
        serveur.GESTION = gardee


def test_missing_model_directory(monde, tmp_path):
    cfg, _ = monde
    import dataclasses
    s = serveur.GESTION.tous["essai"]
    with pytest.raises(RuntimeError, match="no OpenVINO model"):
        s.servable(dataclasses.replace(cfg.engine("llm"), modele=str(tmp_path / "rien")))


def test_server_that_dies_is_reported(monde):
    cfg, _ = monde
    s = serveur.GESTION.tous["essai"]
    engines.chat(cfg.engine("llm"), "un")
    s.proc.kill()
    s.proc.wait()
    time.sleep(0.1)
    # The next request starts it again rather than failing for good.
    assert engines.chat(cfg.engine("llm"), "deux") == "gemma: deux"


def test_unknown_model_restarts_rather_than_edits(monde):
    cfg, store = monde
    import dataclasses
    s = serveur.GESTION.tous["essai"]
    engines.chat(cfg.engine("llm"), "un")
    pid = s.proc.pid
    nouveau = dataclasses.replace(cfg.engine("llm"), model="autre", options="")
    assert engines.chat(nouveau, "deux") == "autre: deux"
    assert s.proc.pid != pid and "autre" in s.config.read_text()


def test_broken_model_left_out_others_served(monde):
    cfg, _ = monde
    import dataclasses
    s = serveur.GESTION.tous["essai"]
    i = [m.model for m in s.moteurs].index("whisper")
    s.moteurs[i] = dataclasses.replace(s.moteurs[i], options="--refuse")
    assert engines.chat(cfg.engine("llm"), "un") == "gemma: un"
    assert "whisper" not in s.servis and "translategemma" in s.servis
