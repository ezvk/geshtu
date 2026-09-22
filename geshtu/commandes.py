"""Voice command matching, and the small desktop bits dictation/commande share.

Port of `voix.py`'s `reference()`/`nombre()`, with the hand-rolled HTTP client
it used for embeddings replaced by `engines.embed()` -- same endpoint shape,
one fewer implementation to keep correct.

⚠️ WHY THIS IS A DAEMON COMMAND AND NOT A Stage/Sink. See geshtu/plugins.py: a
Stage only runs inside `pipeline.process`'s full stage list (diarise,
transcribe, chapter, summarise), and a Sink only runs after all of that. A
spoken command needs exactly one thing -- transcribe, then match -- and
running chapter/summarise on a three-second utterance would be wasted GPU
calls for output nobody asked for. `geshtu/daemon.py`'s `State.commande()`
calls straight into this module instead.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import time

from geshtu import engines

CACHE = pathlib.Path(
    os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")) / "geshtu"
ETAT_DUR = pathlib.Path(
    os.environ.get("XDG_STATE_HOME") or (pathlib.Path.home() / ".local/state")) / "geshtu"


def charge_table() -> tuple[dict, pathlib.Path] | tuple[None, None]:
    """The command table and the path it came from: a user override first,
    then the packaged default -- same precedence as `plugins.py`'s
    local-directory-over-builtin rule. The path travels with the table
    because `reference()` hashes the file to key its cache.

    ⚠️ `/etc/geshtu/commandes.json` IS THE PACKAGED DEFAULT, kept in `/etc`
    rather than baked into the Nix store closure, so it stays editable and
    diffable without a rebuild -- and so the hash-based cache below still
    works against a file that can actually change.
    """
    for candidat in (pathlib.Path.home() / ".config/geshtu/commandes.json",
                     pathlib.Path("/etc/geshtu/commandes.json")):
        if candidat.is_file():
            return json.loads(candidat.read_text()), candidat
    return None, None


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def reference(table: dict, table_path: pathlib.Path, embed_engine) -> dict:
    """The vectors of every formulation, computed ONCE and kept.

    ⚠️ THE CACHE KEY IS A HASH OF THE TABLE FILE: edit a formulation and the
    cache invalidates itself. Without this, a corrected table would keep
    being compared against the old vectors, silently.
    """
    brut = table_path.read_bytes()
    cle = hashlib.sha256(brut).hexdigest()[:16]
    f = CACHE / ("%s.json" % cle)
    if f.is_file():
        return json.loads(f.read_text())
    phrases, ids = [], []
    for c in table["commandes"]:
        for p in c["formulations"]:
            phrases.append(p.replace("{n}", "trois"))
            ids.append(c["id"])
    v = engines.embed(embed_engine, phrases)
    CACHE.mkdir(parents=True, exist_ok=True)
    data = {"ids": ids, "vecs": v}
    f.write_text(json.dumps(data))
    for vieux in CACHE.glob("*.json"):
        if vieux != f:
            vieux.unlink()
    return data


def nombre(texte: str, table: dict) -> str | None:
    """The {n} parameter is READ, never guessed.

    The embedding picks the intent; a number inside it is deterministic code,
    never a second thing asked of a model.
    """
    m = re.search(r"\b(\d+)\b", texte)
    if m:
        return m.group(1)
    for mot, val in table.get("chiffres", {}).items():
        if re.search(r"\b%s\b" % re.escape(mot), texte, re.I):
            return str(val)
    return None


def correspond(texte: str, table: dict, table_path: pathlib.Path,
              embed_engine, seuil: float | None = None) -> dict:
    """Match a transcribed phrase against the table. Always returns a dict;
    `retenu` says whether the match cleared the threshold."""
    ref = reference(table, table_path, embed_engine)
    v = engines.embed(embed_engine, [texte])[0]
    scores = sorted(((_cosine(v, ref["vecs"][i]), ref["ids"][i])
                     for i in range(len(ref["ids"]))), reverse=True)
    seuil = table.get("seuil", 0.75) if seuil is None else seuil
    meilleur_score, meilleur_id = scores[0]
    retenu = meilleur_score >= seuil
    note("commande", texte, meilleur_id, meilleur_score, retenu, scores)
    return {"retenu": retenu, "id": meilleur_id, "score": meilleur_score,
            "prochains": scores[1:4]}


def note(mode: str, texte: str, ident: str, score: float, retenu: bool,
        scores: list[tuple[float, str]] | None = None) -> None:
    """Every recognition, accepted or rejected -- both clouds are needed to
    tune the threshold. `geshtu bilan` reads this file back."""
    f = ETAT_DUR / "commandes.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    ligne = {
        "quand": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode, "texte": texte, "retenu": retenu,
        "meilleur": ident, "score": round(score, 4),
    }
    if scores:
        ligne["suivants"] = [{"id": i, "score": round(sc, 4)} for sc, i in scores[1:4]]
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def dire(titre: str, corps: str = "") -> None:
    """A desktop notification. Never fatal: this is a courtesy, not the
    result -- geshtu's own journal (State.report()) is the real record."""
    try:
        subprocess.run(["noctalia", "msg", "notification-show", titre, corps],
                       timeout=5, check=False)
    except Exception:                                    # noqa: BLE001
        pass


def presse_papier(texte: str) -> None:
    """The transcription always goes to the clipboard, in every mode -- even
    `commande`, even when nothing matched. A rejected command is still
    something you may want to paste and check."""
    try:
        subprocess.run(["noctalia", "msg", "clipboard-copy", texte],
                       timeout=5, check=False)
    except Exception:                                    # noqa: BLE001
        pass


def execute(action: str) -> tuple[bool, str]:
    """Run a matched command's shell action. Returns (ok, detail).

    ⚠️ `MANGO_INSTANCE_SIGNATURE` IS GUARANTEED, NOT HOPED FOR. `mmsg` refuses
    to talk to the compositor without it: "MANGO_INSTANCE_SIGNATURE is not
    set. Did you run 'mmsg' in mango?". `exec-once` children inherit it, but
    nothing guarantees a daemon started as a systemd --user service does --
    and an action that silently does not run looks exactly like a
    recognition failure. ezvk, on voix's own history: "il a compris plein
    ecran il a rien fait". The socket is findable, so it is set here rather
    than assumed.

    ⚠️ THE RESULT IS READ, not fired-and-forgotten. An unwaited Popen made
    every failure invisible -- the notification said "done" regardless, which
    is exactly what made a real failure look like a bad recognition instead.
    """
    env = dict(os.environ)
    if not env.get("MANGO_INSTANCE_SIGNATURE"):
        base = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000"))
        socks = sorted(base.glob("mango-*.sock"))
        if socks:
            env["MANGO_INSTANCE_SIGNATURE"] = str(socks[0])
    r = subprocess.run(["sh", "-c", action], env=env, capture_output=True,
                       text=True, timeout=20)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "").strip()[:120]
    return True, ""


def bilan() -> None:
    """Print score distributions for accepted vs rejected recognitions, and
    the closest-scoring rejections -- the tool the threshold was calibrated
    with, ported so it stays usable."""
    f = ETAT_DUR / "commandes.jsonl"
    if not f.is_file():
        print("aucun historique -- ~/.local/state/geshtu/commandes.jsonl absent")
        return
    lignes = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    retenues = [x for x in lignes if x["retenu"]]
    rejetees = [x for x in lignes if not x["retenu"]]
    print("  retenues : %d" % len(retenues))
    if retenues:
        scores = [x["score"] for x in retenues]
        print("    scores : %.3f - %.3f" % (min(scores), max(scores)))
    print("  rejetees : %d" % len(rejetees))
    if rejetees:
        scores = [x["score"] for x in rejetees]
        print("    scores : %.3f - %.3f" % (min(scores), max(scores)))
    print("  rejets les plus proches du seuil :")
    for x in sorted(rejetees, key=lambda x: -x["score"])[:15]:
        print("    %.3f  %-22s « %s »"
              % (x["score"], x["meilleur"][:22], x["texte"][:50]))
