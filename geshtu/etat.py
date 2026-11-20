"""What the daemon is doing, as ONE named mode, and what the main toggle does.

⚠️ IT USED TO BE WORKED OUT THREE TIMES, THREE WAYS. The window, the tray icon
and `geshtu status` each rebuilt "what is going on" from the raw status
fields, with their own order of precedence:

  - the CLI ignored `dictating` and `commanding`, and said « idle » while the
    microphone was being captured;
  - the tray once tested `recording` alone, and its middle click STARTED two
    meetings during a dictation (ezvk, 2026-10-08, see
    daemon._abandonne_courte);
  - the window and the tray tested `processing` BEFORE `recording`: a
    recording started while the previous one was still being transcribed
    could not be stopped from either -- the daemon would have stopped it.

The README's rule is that the daemon holds the state and clients stay thin.
So the daemon names the mode here, once, and sends it with every `status`;
clients show it and act on it, they do not derive it.

Precedence, highest first -- the order the clients already used:

    recording    a meeting is being captured        toggle: stop (and transcribe)
    dictating    a dictation is being captured      toggle: discard
    commanding   a voice command is being captured  toggle: discard
    processing   a stage runs, nothing is captured  toggle: wait (nothing to do)
    idle         nothing at all                     toggle: start

A meeting and a dictation may run at once (daemon.State: separate slots); the
meeting wins, as it did in every client.
"""
from __future__ import annotations

MODES = ("recording", "dictating", "commanding", "processing", "idle")

ACTIONS = {
    "recording": "stop",
    "dictating": "discard",
    "commanding": "discard",
    "processing": "wait",
    "idle": "start",
}

# What capturing the microphone or a stream looks like from outside: the
# tray's red dot, NeedsAttention.
CAPTURE = ("recording", "dictating", "commanding")


def mode(recording: bool, dictating: bool, commanding: bool, processing: bool) -> str:
    if recording:
        return "recording"
    if dictating:
        return "dictating"
    if commanding:
        return "commanding"
    if processing:
        return "processing"
    return "idle"


def depuis(status: dict) -> tuple[str, str]:
    """(mode, toggle action) from a `status` answer.

    A daemon older than this module sends no `mode`: it is then derived from
    the raw fields with the same function, so a window opened before the
    daemon restarts behaves exactly as one opened after.
    """
    m = status.get("mode")
    if m not in ACTIONS:
        m = mode(bool(status.get("recording")), bool(status.get("dictating")),
                 bool(status.get("commanding")), bool(status.get("processing")))
    return m, ACTIONS[m]
