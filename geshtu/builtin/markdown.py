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
import time


class MarkdownSink:
    name = "markdown"

    def deliver(self, session, cfg, report) -> None:
        # ⚠️ NO NOTE FOR A SESSION WITH NOTHING IN IT. A recording that caught
        # only silence produces a file with a heading and an empty transcript,
        # which is indistinguishable at a glance from a real note and quietly
        # fills the folder. The daemon log already said why it was silent;
        # repeating it as a document helps nobody.
        if not session.segments:
            report("nothing was transcribed -- no note written")
            return
        out = pathlib.Path(
            cfg.raw.get("markdown", {}).get("folder")
            or (pathlib.Path.home() / "Documents" / "geshtu")).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d", time.localtime(session.started))
        # The session id already begins with the date; repeating it gives
        # names like 2026-09-21-2026-09-21-11-12.md.
        name = session.title or session.id.removeprefix(day).strip("_-") or "session"
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name)[:60].strip("-").lower()
        base = "%s-%s" % (day, slug or "session")
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
        for s in session.segments:
            qui = ("**%s** " % s.speaker) if s.speaker else ""
            lines.append("%s`%s` %s" % (qui, _hms(s.start), s.text))
            lines.append("")
        transcription.write_text("\n".join(lines), encoding="utf-8")
        report("written %s" % transcription)

        # ---- the summary: only when there is one ---------------------------
        if not a_resume:
            report("no summary -- only the transcript was written")
            return
        lines = _entete(session) + [
            "# %s" % titre, "",
            "Transcription complète : [%s](%s)" % (transcription.name, transcription.name), "",
        ]
        if session.summary:
            lines += [session.summary, ""]
        for c in chapitres:
            lines += ["## %s  *(%s)*" % (c.title, _hms(c.start)), "", c.body, ""]
        resume.write_text("\n".join(lines), encoding="utf-8")
        report("written %s" % resume)


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
