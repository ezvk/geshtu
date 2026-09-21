"""Write the session as a dated Markdown note."""
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
        path = out / ("%s-%s.md" % (day, slug or "session"))

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
        lines += ["language: %s" % session.language, "---", ""]
        lines += ["# %s" % (session.title or session.id), ""]
        if session.summary:
            lines += [session.summary, ""]
        for c in session.chapters:
            if not c.title:
                continue
            lines += ["## %s  *(%s)*" % (c.title, _hms(c.start)), "", c.body, ""]

        # ⚠️ THE FULL TRANSCRIPT IS KEPT, folded at the end. A summary cannot
        # be checked against itself, and this is exactly what is needed the day
        # the model invents something.
        lines += ["---", "", "<details><summary>Full transcript</summary>", ""]
        for s in session.segments:
            lines.append("**%s** %s" % (_hms(s.start), s.text))
            lines.append("")
        lines += ["</details>", ""]

        path.write_text("\n".join(lines), encoding="utf-8")
        report("written %s" % path)


def _hms(seconds: float) -> str:
    s = int(seconds)
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


PLUGIN = MarkdownSink
