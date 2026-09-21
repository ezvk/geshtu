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
            # ⚠️ LES LOCUTEURS ENTRENT DANS L INVITE quand on les connait :
            # sans eux le modele ne peut pas dire QUI s engage sur quoi, et
            # une action sans responsable ne vaut pas grand chose dans un
            # compte rendu.
            morceaux = []
            for s in session.segments:
                if s.start < chapter.start or s.end > chapter.end:
                    continue
                morceaux.append("%s: %s" % (s.speaker, s.text) if s.speaker else s.text)
            text = "\n".join(morceaux).strip()
            if len(text.split()) < MIN_WORDS:
                report("  chapter %d skipped: %d words" % (i, len(text.split())))
                continue
            context = ("What came before, for reference only:\n%s\n\n" % digest
                       if digest else "")
            try:
                answer, coupe = _demande(engine, words["chapter"] % (text, context),
                                         words["system"])
            except Exception as exc:                     # noqa: BLE001
                # ⚠️ UN CHAPITRE PERDU NE DOIT PAS COUTER LA REUNION. Vecu le
                # 2026-09-21 : le chapitre 2 depassait le budget de jetons et
                # l exception a emporte les trois, plus le compte rendu.
                report("  chapter %d failed: %s" % (i, exc))
                continue
            chapter.title, chapter.body = _title_and_body(answer)
            if coupe:
                chapter.body += "\n\n*(réponse tronquée par le modèle)*"
            done.append(chapter)
            digest = (digest + " " + chapter.title)[-400:]
            report("  chapter %d/%d: %s%s"
                   % (i, len(session.chapters), chapter.title,
                      "  [tronqué]" if coupe else ""))

        if not done:
            report("nothing usable was said -- no summary produced")
            return
        joined = "\n\n".join("%s\n%s" % (c.title, c.body) for c in done)
        try:
            session.summary, coupe = _demande(engine, words["synthesis"] % joined,
                                              words["system"], budget=1400)
            if coupe:
                session.summary += "\n\n*(réponse tronquée par le modèle)*"
        except Exception as exc:                         # noqa: BLE001
            report("overall summary failed: %s" % exc)
        session.title = done[0].title


def _demande(engine, prompt, system, budget=1400):
    """Demande une reponse, en agrandissant le budget plutot qu en perdant tout.

    ⚠️ TROIS TENTATIVES, ET LA DERNIERE ACCEPTE LA TRONCATURE — marquee comme
    telle. Refuser une reponse coupee est juste : on n en tire aucune
    conclusion sure. Mais refuser au point de ne RIEN rendre transforme un
    chapitre bavard en reunion perdue, ce qui est pire. Le lecteur doit
    pouvoir voir que le modele a ete coupe, pas deviner pourquoi il manque un
    chapitre.
    """
    for maxi in (budget, budget * 2):
        try:
            return engines.chat(engine, prompt, system, max_tokens=maxi), False
        except RuntimeError as exc:
            if "truncated" not in str(exc):
                raise
    return engines.chat(engine, prompt, system, max_tokens=budget * 2,
                        allow_truncated=True), True


def _title_and_body(raw: str) -> tuple[str, str]:
    """⚠️ TOLERANT ON PURPOSE. A model asked for "TITLE:" will write "**TITLE
    :**", "## Titre -", or drop the label entirely. A strict parser produced a
    whole report of chapters called "(untitled)".
    """
    lines = [x.strip() for x in raw.splitlines()]
    title, body = "", []
    # ⚠️ THE LABEL IS MATCHED IN EVERY LANGUAGE THIS PROMPTS IN, and around
    # whatever decoration the model adds. A French-only pattern let an English
    # run through untouched and the heading came out as
    # "# TITLE:** Secret Shared About ...". A model asked for "TITLE:" will
    # write "**TITLE:**", "## Title -", or drop the label entirely; a strict
    # parser once produced a whole report of chapters called "(untitled)".
    label = re.compile(r"(?i)^[*#\s]*(titre|title)\s*[:\-]?\s*[*#\s]*")
    for line in lines:
        if not title and label.match(line) and label.sub("", line).strip(" *#:-"):
            title = label.sub("", line).strip(" *#:-")
            continue
        body.append(line.strip("*# ") if line.strip("*# ").isupper() else line)
    if not title:
        title = next((x for x in lines if x), "").strip(" *#:-")[:70]
        body = lines[1:]
    return title, "\n".join(body).strip()


PLUGIN = Summarise
