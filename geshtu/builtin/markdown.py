"""Write the session as two dated Markdown notes: the summary, and the full
transcript.

⚠️ TWO FILES, NOT ONE. ezvk, 2026-09-27: "geshtu need to produce two files the
resume and the full transcript, not both in the same one". The transcript used
to be folded at the end of the summary note; it now lives beside it, and each
note links to the other.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import time
import unicodedata


class MarkdownSink:
    name = "markdown"

    def deliver(self, session, cfg, report) -> None:
        # ⚠️ NO NOTE FOR A SESSION WITH NOTHING IN IT. A recording that caught
        # only silence produces a file with a heading and an empty transcript,
        # which is indistinguishable at a glance from a real note and quietly
        # fills the folder. The daemon log already said why it was silent;
        # repeating it as a document helps nobody.
        # A session with shared screens and no speech (a silent demo, a muted
        # call) still has something to keep: measured on utu, 2026-11-20, six
        # screens read and no note at all.
        if not session.segments and not session.ecrans:
            report("nothing was transcribed -- no note written")
            return
        out, base = emplacement(session, cfg)
        out.mkdir(parents=True, exist_ok=True)
        resume = out / (base + "-resume.md")
        transcription = out / (base + "-transcription.md")
        titre = session.title or session.id

        # ---- the full transcript: ALWAYS written when something was said ----
        # ⚠️ IT IS KEPT WHOLE, and it comes first. A summary cannot be checked
        # against itself, and this is exactly what is needed the day the model
        # invents something -- or the day the summary step failed (the LLM is
        # on ishtar: off the tailnet there is a transcript and no summary).
        chapitres = [c for c in session.chapters if c.title]
        a_resume = bool(session.summary or chapitres)
        lines = _entete(session) + ["# %s — transcription" % titre, ""]
        # Link only to a note that exists: no summary, no dangling link.
        if a_resume:
            lines += ["Résumé : [%s](%s)" % (resume.name, resume.name), ""]
        lines += _lien_audio(session, out)
        ecrans = _copie_ecrans(session, out, base)
        # The screens go IN the transcript, where they were shown: what was
        # said about a slide is read beside the slide.
        a_placer = list(ecrans)
        for s in session.segments:
            while a_placer and a_placer[0][0].t <= s.start:
                lines += _bloc_ecran(*a_placer.pop(0))
            qui = ("**%s** " % session.nom(s.speaker)) if s.speaker else ""
            lines.append("%s`%s` %s" % (qui, _hms(s.start), s.text))
            lines.append("")
        for e in a_placer:
            lines += _bloc_ecran(*e)
        transcription.write_text("\n".join(lines), encoding="utf-8")
        report("written %s" % transcription)

        # ---- the summary: only when there is one ---------------------------
        if not a_resume:
            report("no summary -- only the transcript was written")
            _retire_anciennes(session, [transcription], (resume, transcription), report)
            return
        lines = _entete(session) + [
            "# %s" % titre, "",
            "Transcription complète : [%s](%s)" % (transcription.name, transcription.name), "",
        ] + _lien_audio(session, out)
        if session.summary:
            lines += [session.summary, ""]
        for c in chapitres:
            lines += ["## %s  *(%s)*" % (c.title, _hms(c.start)), "", c.body, ""]
        # ⚠️ THE SCREENS ARE MATERIAL FOR THE FINAL DOCUMENT (ezvk, 2026-11-20:
        # « matière à insérer dans doc final »): each one whole, picture first,
        # so it can be lifted as it is into a report.
        if ecrans:
            lines += ["## Écrans partagés", ""]
            for e in ecrans:
                lines += _bloc_ecran(*e, citation=False)
        resume.write_text("\n".join(lines), encoding="utf-8")
        report("written %s" % resume)
        _retire_anciennes(session, [transcription, resume], (resume, transcription), report)


def _copie_ecrans(session, out: pathlib.Path, base: str) -> list:
    """The kept screens, copied beside the notes (<base>-ecrans/), as
    (Ecran, relative link) pairs. A note must not point into the session
    folder: that one is internal and goes when the session is cleaned."""
    if not session.ecrans:
        return []
    dossier = out / (base + "-ecrans")
    dossier.mkdir(parents=True, exist_ok=True)
    pris = []
    for e in session.ecrans:
        if not e.image.exists():
            continue
        cible = dossier / e.image.name
        if not cible.exists() or cible.stat().st_size != e.image.stat().st_size:
            shutil.copy2(e.image, cible)
        pris.append((e, "%s/%s" % (dossier.name, cible.name)))
    return pris


def _bloc_ecran(e, lien: str, citation: bool = True) -> list[str]:
    tete = "🖥️ **Écran partagé** `%s`" % _hms(e.t)
    corps = [tete, "", "![%s](%s)" % (_hms(e.t), lien), ""]
    if e.texte:
        # Line by line: the text is several lines, and each needs its own "> "
        # (measured on utu, 2026-11-20: only the first one stayed in the quote).
        # A heading the model wrote (« ## Contenu de l'écran ») would cut the
        # note's own sections, so it becomes bold text.
        for ligne in e.texte.strip().splitlines():
            titre = re.match(r"\s*#{1,6}\s+(.*)", ligne)
            corps.append("**%s**" % titre.group(1).strip("* ") if titre else ligne)
        corps.append("")
    if not citation:
        return corps
    return [("> " + ligne) if ligne else ">" for ligne in corps[:-1]] + [""]


def _retire_anciennes(session, ecrites, noms_actuels, report) -> None:
    """⚠️ THE NAME FOLLOWS THE TITLE, AND THE TITLE ARRIVES LATE. A meeting is
    first written as 2026-11-05-14-02-transcription.md; once summarised it is
    titled, and written again under the title. The first note stayed behind,
    the same transcript twice in the folder (ezvk, 2026-11-05 : « corrige le
    doublon dans geshtu »).

    Only a note THIS sink wrote for THIS session under ANOTHER name goes: one
    whose name is still current is never removed (a rewrite without summary
    keeps the summary note), and one that no longer carries this session's id
    in its header -- replaced, or reused by someone -- is left alone.
    """
    for p in session.notes:
        if p in noms_actuels or not p.is_file():
            continue
        if _id_de(p) != session.id:
            continue
        p.unlink()
        report("old note removed: %s" % p.name)
        # Its screens went with it: <base>-ecrans/ beside the old name
        # (measured on utu, 2026-11-20: the folder stayed after the rename).
        ancien = _dossier_ecrans(p)
        if ancien and ancien not in {_dossier_ecrans(n) for n in noms_actuels} \
                and ancien.is_dir():
            shutil.rmtree(ancien)
            report("old screens removed: %s" % ancien.name)
    session.notes = list(ecrites)


def _dossier_ecrans(note: pathlib.Path) -> pathlib.Path | None:
    for fin in ("-transcription.md", "-resume.md"):
        if note.name.endswith(fin):
            return note.with_name(note.name[:-len(fin)] + "-ecrans")
    return None


def _id_de(note: pathlib.Path) -> str | None:
    try:
        with note.open(encoding="utf-8") as f:
            for _, line in zip(range(4), f):
                if line.startswith("id: "):
                    return line[4:].strip()
    except OSError:
        pass
    return None


def emplacement(session, cfg) -> tuple[pathlib.Path, str]:
    """The notes folder and the dated base name of this session's files --
    shared with the mp3 sink, so the audio sits beside its note."""
    out = pathlib.Path(
        cfg.raw.get("markdown", {}).get("folder")
        or (pathlib.Path.home() / "Documents" / "geshtu")).expanduser()
    day = time.strftime("%Y-%m-%d", time.localtime(session.started))
    # The session id already begins with the date; repeating it gives
    # names like 2026-09-21-2026-09-21-11-12.md.
    name = session.title or session.id.removeprefix(day).strip("_-") or "session"
    # ⚠️ ACCENTS ARE FOLDED, NOT DROPPED: « énergétique » became « nerg-tique »
    # (2026-11-20). The accented letter becomes its base letter first.
    plat = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", plat)[:60].strip("-").lower()
    return out, "%s-%s" % (day, slug or "session")


def _lien_audio(session, out: pathlib.Path) -> list[str]:
    # Link only to a file that exists, and only when it sits beside the note.
    if session.audio and session.audio.exists() and session.audio.parent == out:
        return ["Audio : [%s](%s)" % (session.audio.name, session.audio.name), ""]
    return []


def _entete(session) -> list[str]:
    lines = [
        "---",
        "id: %s" % session.id,
        "date: %s" % time.strftime("%Y-%m-%d %H:%M", time.localtime(session.started)),
        "duration: %d" % int(session.duration),
        "sources:",
    ]
    for t in session.tracks:
        lines.append("  - %s: %s%s" % (t.kind, t.source,
                                       "   # SILENT" if t.silent else ""))
    return lines + ["language: %s" % session.language, "---", ""]


def _hms(seconds: float) -> str:
    s = int(seconds)
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


PLUGIN = MarkdownSink
