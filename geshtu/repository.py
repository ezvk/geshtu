"""Dropping a model in a folder and having it served.

ezvk: "les modeles doivent etre testables par ajout simple dans un dossier".

The model server does the work; this module only knows the three calls and
the traps. Verified against OpenVINO Model Server 2026.3.1 on the machine:

    ovms --configure --model_path P --task T [...]   writes graph.pbtxt in P
    ovms --add_to_config --config_path C --model_path P --model_name N
    ovms --remove_from_config --config_path C --model_name N

⚠️ THE SERVER MUST BE WATCHING. A model added to a config file changes
nothing until the server re-reads it. Start it with

    ovms --config_path <C> --file_system_poll_wait_seconds 5

and the drop-in is genuinely a drop-in. Started with `--model_path`, it
serves exactly one model and no amount of editing a config it never reads
will change that -- which looks, from the outside, exactly like a config
that is being ignored.

⚠️ `--configure` WRITES INTO THE MODEL DIRECTORY. graph.pbtxt lands next to
the weights, so the directory must be writable, and pointing this at a
read-only or shared model store fails in a way that reads like a permission
bug rather than a design decision. Measured on a real model: it appeared in
the live directory of a running single-model server.

⚠️ AND A GENERATIVE MODEL NEEDS ITS TASK. Without `--task`, a config entry
for an LLM yields a server that starts and then answers nothing useful: the
graph that turns an HTTP payload into generation is exactly what --configure
builds.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

TASKS = ("text_generation", "embeddings", "rerank", "speech2text",
         "image_generation")


def _ovms(cfg) -> str:
    binary = cfg.raw.get("repository", {}).get("ovms", "ovms")
    found = shutil.which(binary) or binary
    return found


def config_path(cfg) -> pathlib.Path:
    raw = cfg.raw.get("repository", {}).get("config")
    if not raw:
        raise SystemExit(
            "no [repository] config declared -- see docs/models.md")
    return pathlib.Path(raw).expanduser()


def models_path(cfg) -> pathlib.Path:
    raw = cfg.raw.get("repository", {}).get("models")
    if not raw:
        raise SystemExit(
            "no [repository] models directory declared -- see docs/models.md")
    return pathlib.Path(raw).expanduser()


def _run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def pull(cfg, source: str, name: str, task: str, options: list[str]) -> str:
    """Fetch a model from Hugging Face into the repository."""
    repo = models_path(cfg)
    repo.mkdir(parents=True, exist_ok=True)
    code, out = _run([_ovms(cfg), "--pull", "--source_model", source,
                      "--model_repository_path", str(repo),
                      "--model_name", name, "--task", task] + options)
    if code:
        raise SystemExit("pull failed:\n" + out)
    return str(repo / name)


def add(cfg, path: str, name: str, task: str, options: list[str]) -> str:
    """Prepare a local model directory and list it in the served config."""
    model = pathlib.Path(path).expanduser().resolve()
    if not model.is_dir():
        raise SystemExit("not a directory: %s" % model)
    if task:
        if task not in TASKS:
            raise SystemExit("unknown task %r (known: %s)"
                             % (task, ", ".join(TASKS)))
        code, out = _run([_ovms(cfg), "--configure", "--model_path", str(model),
                          "--task", task] + options)
        if code:
            raise SystemExit("configure failed:\n" + out)
    conf = config_path(cfg)
    conf.parent.mkdir(parents=True, exist_ok=True)
    code, out = _run([_ovms(cfg), "--add_to_config",
                      "--config_path", str(conf),
                      "--model_path", str(model),
                      "--model_name", name])
    if code:
        raise SystemExit("add_to_config failed:\n" + out)
    return out


def remove(cfg, name: str) -> str:
    conf = config_path(cfg)
    code, out = _run([_ovms(cfg), "--remove_from_config",
                      "--config_path", str(conf), "--model_name", name])
    if code:
        raise SystemExit("remove_from_config failed:\n" + out)
    return out


def scan(cfg) -> list[dict]:
    """What is sitting in the repository directory, served or not.

    ⚠️ PRESENT IS NOT SERVED. A directory of weights nobody listed in the
    config is invisible to the server, and `geshtu models` -- which asks the
    SERVER -- will not show it. Both views are needed, and conflating them is
    how you spend an afternoon wondering why a model you can see is missing.
    """
    repo = models_path(cfg)
    if not repo.is_dir():
        return []
    out = []
    for d in sorted(repo.iterdir()):
        if not d.is_dir():
            continue
        weights = list(d.rglob("*.xml")) or list(d.rglob("*.gguf"))
        out.append({"name": d.name, "path": str(d),
                    "weights": bool(weights),
                    "graph": (d / "graph.pbtxt").is_file()})
    return out
