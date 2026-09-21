"""Summarising, one chapter at a time, then a synthesis of the summaries.

⚠️ NEVER SUBMIT THE WHOLE TRANSCRIPT. See chapters.py: the accelerator refuses
any prompt above 8192 tokens and an hour of speech is well past that. The
cascade is not a fallback here, it is the architecture.

⚠️ AND THE CASCADE COSTS SOMETHING, WHICH THE OUTPUT SHOULD NOT HIDE: it loses
the callbacks that span the whole meeting ("going back to what Marie said
earlier"), which are often the most valuable part of a report. Each chapter
therefore carries a short running digest of the ones before it.
"""
from __future__ import annotations

import re

from geshtu import engines

MIN_WORDS = 40          # below this a chapter has nothing to summarise

PROMPTS = {
    "en": {
        "system": "You summarise meeting transcripts. Be factual. "
                  "Never invent a decision that was not stated.",
        "chapter": ("Transcript of one part of a meeting:\n\n%s\n\n"
                    "%s"
                    "Write:\nTITLE: a short title, under eight words\n"
                    "then a paragraph of what was said, then, only if they "
                    "were actually stated, the lines DECISIONS: and ACTIONS: "
                    "with one bullet each."),
        "synthesis": ("Chapter summaries of a meeting:\n\n%s\n\n"
                      "Write a short overall summary, then the decisions and "
                      "the actions with their owner and deadline when stated. "
                      "Omit a section entirely rather than filling it."),
    },
    "fr": {
        "system": "Tu résumes des transcriptions de réunion. Sois factuel. "
                  "N'invente jamais une décision qui n'a pas été énoncée.",
        "chapter": ("Transcription d'une partie de réunion :\n\n%s\n\n"
                    "%s"
                    "Écris :\nTITRE : un titre court, moins de huit mots\n"
                    "puis un paragraphe de ce qui a été dit, puis, seulement "
                    "si elles ont réellement été énoncées, les lignes "
                    "DÉCISIONS : et ACTIONS : avec une puce chacune."),
        "synthesis": ("Résumés des chapitres d'une réunion :\n\n%s\n\n"
                      "Écris un résumé général court, puis les décisions et "
                      "les actions avec leur responsable et leur échéance "
                      "quand ils sont dits. Omets une section entière plutôt "
                      "que de la remplir."),
    },
}


class Summarise:
    name = "summarise"
    requires = ("chapters",)
    provides = ("summary",)

    def run(self, session, cfg, report) -> None:
        if not session.chapters:
            report("no chapters to summarise")
            return
        engine = cfg.engine("llm")
        words = PROMPTS.get(session.language, PROMPTS["en"])
        digest = ""
        done = []

        for i, chapter in enumerate(session.chapters, 1):
            text = " ".join(s.text for s in session.segments
                            if s.start >= chapter.start and s.end <= chapter.end).strip()
            if len(text.split()) < MIN_WORDS:
                report("  chapter %d skipped: %d words" % (i, len(text.split())))
                continue
            context = ("What came before, for reference only:\n%s\n\n" % digest
                       if digest else "")
            answer = engines.chat(engine, words["chapter"] % (text, context),
                                  words["system"], max_tokens=1100)
            chapter.title, chapter.body = _title_and_body(answer)
            done.append(chapter)
            digest = (digest + " " + chapter.title)[-400:]
            report("  chapter %d/%d: %s" % (i, len(session.chapters), chapter.title))

        if not done:
            report("nothing usable was said -- no summary produced")
            return
        joined = "\n\n".join("%s\n%s" % (c.title, c.body) for c in done)
        session.summary = engines.chat(engine, words["synthesis"] % joined,
                                       words["system"], max_tokens=1200)
        session.title = done[0].title


def _title_and_body(raw: str) -> tuple[str, str]:
    """⚠️ TOLERANT ON PURPOSE. A model asked for "TITLE:" will write "**TITLE
    :**", "## Titre -", or drop the label entirely. A strict parser produced a
    whole report of chapters called "(untitled)".
    """
    lines = [x.strip().strip("*# ") for x in raw.splitlines()]
    title, body = "", []
    for line in lines:
        if not title and re.match(r"(?i)^titre?\s*:", line):
            title = re.sub(r"(?i)^titre?\s*:\s*", "", line).strip(" *:")
            continue
        body.append(line)
    if not title:
        title = next((x for x in lines if x), "")[:70]
        body = lines[1:]
    return title, "\n".join(body).strip()


PLUGIN = Summarise
