"""Write the session as a dated Markdown note."""
from __future__ import annotations

import pathlib
import re
import time


class MarkdownSink:
    name = "markdown"

    def deliver(self, session, cfg, report) -> None:
        out = pathlib.Path(
            cfg.raw.get("markdown", {}).get("folder")
            or (pathlib.Path.home() / "Documents" / "geshtu")).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", session.title or session.id)[:60].strip("-")
        path = out / ("%s-%s.md" % (time.strftime("%Y-%m-%d", time.localtime(session.started)),
                                    slug.lower() or "session"))

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
