import itertools

import pytest

from geshtu import etat


def test_every_mode_has_an_action():
    assert set(etat.MODES) == set(etat.ACTIONS)


@pytest.mark.parametrize("rec,dic,com,proc,attendu", [
    (False, False, False, False, "idle"),
    (False, False, False, True, "processing"),
    (False, True, False, False, "dictating"),
    (False, False, True, False, "commanding"),
    (True, False, False, False, "recording"),
    # A meeting recorded while the previous one is processed: stop must work.
    (True, False, False, True, "recording"),
    # A dictation over a meeting: the meeting wins, as in every client.
    (True, True, False, False, "recording"),
    # A dictation while a meeting is processed: it can still be discarded.
    (False, True, False, True, "dictating"),
])
def test_precedence(rec, dic, com, proc, attendu):
    assert etat.mode(rec, dic, com, proc) == attendu


def test_recording_while_processing_can_be_stopped():
    assert etat.depuis({"recording": "2026-11-08_10-00",
                        "processing": "2026-11-08_09-00"}) == ("recording", "stop")


def test_short_captures_are_discarded_not_started():
    assert etat.depuis({"dictating": True}) == ("dictating", "discard")
    assert etat.depuis({"commanding": True, "processing": "x"}) == ("commanding", "discard")


COMBINAISONS = list(itertools.product([None, "s"], [False, True], [False, True], ["", "s"]))


@pytest.mark.parametrize("champs", COMBINAISONS)
def test_old_daemon_without_mode_gives_the_same_answer(champs):
    rec, dic, com, proc = champs
    brut = {"recording": rec, "dictating": dic, "commanding": com, "processing": proc}
    m = etat.mode(bool(rec), dic, com, bool(proc))
    assert etat.depuis(brut) == (m, etat.ACTIONS[m])
    assert etat.depuis(dict(brut, mode=m, action=etat.ACTIONS[m])) == (m, etat.ACTIONS[m])


def test_unknown_mode_falls_back_to_the_fields():
    assert etat.depuis({"mode": "bogus", "recording": "s"}) == ("recording", "stop")


def test_capture_modes():
    assert set(etat.CAPTURE) == {"recording", "dictating", "commanding"}
