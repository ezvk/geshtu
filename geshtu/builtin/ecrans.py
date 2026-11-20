"""The shared screen, read: what each kept screen says, for the notes.

The pictures come from the live capture (geshtu/ecran.py) or, for a session
made from a video file, are found in that file here. Each is then read by the
`vision` engine -- an omni or vision model behind /chat/completions.

⚠️ NO FIGURE IS TAKEN FROM A CHART. Measured with LFM3-Omni-2B (2026-11-20):
the text of a slide is copied faithfully, but the values « read » on a box plot
or a line chart are invented -- a full table of them, with or without the
grounding prompt of the model card. The prompt therefore asks what a chart
SHOWS, never its numbers; the picture itself goes in the note for the rest.
"""
from __future__ import annotations

import re

from geshtu import ecran, engines

# LFM3-Omni's card: « To ground answers in the image, use the system prompt
# `Answer questions about the provided image. Base every answer only on what
# can be seen.` » -- and `Respond in text.` keeps an omni model from speaking.
SYSTEME = ("Answer questions about the provided image. Base every answer only "
           "on what can be seen. Respond in text.")

DEMANDES = {
    "fr": ("Recopie le contenu de cet écran pour un compte rendu : son titre, puis "
           "son texte sous forme de liste. S'il contient un tableau, recopie-le en "
           "tableau Markdown. S'il contient un graphique ou un schéma, dis en une "
           "phrase ce qu'il représente, sans donner aucun chiffre. Ne recopie que "
           "ce qui est écrit."),
    "en": ("Copy the content of this screen for meeting notes: its title, then its "
           "text as a list. If it holds a table, copy it as a Markdown table. If it "
           "holds a chart or a diagram, say in one sentence what it shows, without "
           "giving any figure. Copy only what is written."),
}


class Ecrans:
    name = "ecrans"
    requires = ()
    provides = ("ecrans",)

    def run(self, session, cfg, report) -> None:
        reglages = ecran.Reglages.depuis(cfg.raw.get("ecran"))
        if not session.ecrans and session.video and session.video.exists():
            for t, image in ecran.dans_video(session.video, session.root / "ecrans",
                                             reglages, report):
                session.ecrans.append(ecran._ecran(t, image))
        if not session.ecrans:
            report("no screen to read")
            return
        engine = cfg.engines.get("vision")
        if engine is None:
            report("%d screens kept, not read: no [engines.vision]" % len(session.ecrans))
            return
        demande = DEMANDES.get(session.language, DEMANDES["en"])
        for k, e in enumerate(session.ecrans, 1):
            if e.texte or not e.image.exists():
                continue
            try:
                e.texte = engines.read_image(engine, e.image, demande, SYSTEME)
            except Exception as exc:                     # noqa: BLE001
                # One unreadable screen must not cost the others.
                report("  screen %d/%d failed: %s" % (k, len(session.ecrans), exc))
                continue
            report("  screen %d/%d read" % (k, len(session.ecrans)))
        avant = len(session.ecrans)
        session.ecrans = fusionne(session.ecrans)
        if len(session.ecrans) < avant:
            report("screens: %d kept, %d repeats merged"
                   % (len(session.ecrans), avant - len(session.ecrans)))


def _mots(texte: str) -> set[str]:
    return set(re.findall(r"\w+", texte.lower()))


def fusionne(liste: list) -> list:
    """Two screens in a row that say the same thing are one screen: the fuller
    one is kept, since a slide that is revealed bullet by bullet is most
    complete last.

    ⚠️ THIS IS WHERE THE DETECTOR'S REPEATS GO. The detector is set to never
    miss a revealed bullet (measured: 0 slide state lost of 29), and the price
    is a repeat each time a speaker who stood still for a minute moves again.
    Judged on what the screen SAYS, a repeat is plain: (nearly) all the words
    of one are in the other, and the one that says more is kept.
    """
    out = []
    for e in liste:
        if out and out[-1].texte and e.texte:
            a, b = _mots(out[-1].texte), _mots(e.texte)
            # Either way round: the same title slide read twice comes back
            # once with its banner, once without (measured, 2026-11-20).
            if a and b and len(a & b) / min(len(a), len(b)) >= 0.9:
                garde, jete = (out[-1], e) if len(a) > len(b) else (e, out[-1])
                out[-1] = garde
                if jete.image != garde.image and jete.image.exists():
                    jete.image.unlink()
                continue
        out.append(e)
    return out


PLUGIN = Ecrans
