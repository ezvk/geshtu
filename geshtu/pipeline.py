"""Running the stages, then the sinks.

Deliberately dumb: it resolves names to plugins, runs them in the configured
order, and saves the session after each one. All the intelligence lives in the
stages, which is what lets a user reorder them in a config file.
"""
from __future__ import annotations

import traceback
import typing

from geshtu import plugins

if typing.TYPE_CHECKING:
    from geshtu.config import Config
    from geshtu.models import Session


def process(session: "Session", cfg: "Config", report=print) -> bool:
    """Run every configured stage, then every sink. Returns False on failure.

    ⚠️ THE SESSION IS SAVED AFTER EACH STAGE. A crash in summarisation must
    never cost the transcription that preceded it: audio is the only thing in
    here that cannot be recomputed, and transcription is the only thing that
    takes real time.
    """
    available = plugins.stages()
    for name in cfg.stages:
        stage = available.get(name)
        if stage is None:
            report("stage %r is not installed -- skipped" % name)
            continue
        report("stage: %s" % name)
        try:
            stage().run(session, cfg, report)
        except Exception:                                # noqa: BLE001
            report("stage %s failed:\n%s" % (name, traceback.format_exc()))
            session.save()
            return False
        session.save()

    outputs = plugins.sinks()
    for name in cfg.sinks:
        sink = outputs.get(name)
        if sink is None:
            report("sink %r is not installed -- skipped" % name)
            continue
        try:
            sink().deliver(session, cfg, report)
        except Exception:                                # noqa: BLE001
            # ⚠️ A sink failure is NOT a pipeline failure. The transcript and
            # the summary already exist; a Matrix outage must not make the run
            # look lost.
            report("sink %s failed: %s" % (name, traceback.format_exc(limit=1)))
    session.save()
    return True
