"""Guard against mango's key-repeat storm on a toggle binding.

Port of `voix.py`'s `repetition_clavier()`, parameterised by command name
because geshtu carries several toggles (`dictee`, `commande`) where `voix`
only ever had one per process.

⚠️ NOT A THEORETICAL RISK, A MEASURED ONE. ezvk, 2026-09-16: "lancement en
cascade", then "je dois laisser appuye ?". Holding the key was a natural
gesture -- the walkie-talkie reflex -- and mango RELAUNCHES THE BINDING ON
EVERY REPEAT. Verified in the source, mango 0.16.2 `src/input/keyboard.h`,
function `keyrepeat()`:

    for (i = 0; i < group->nsyms; i++)
        keybinding(WL_KEYBOARD_KEY_STATE_PRESSED, ...);

Since a toggle inverts state on every call, holding the key flips it
start/stop/start/stop as fast as the repeat rate allows.

⚠️ THREE WRONG VERSIONS BEFORE THIS ONE, all load-bearing lessons:

  v1 -- a flat one-second threshold, refreshed only on ACCEPTED calls.
  Holding the key past one second still flipped it, just at 1 Hz instead of
  25. Less spectacular, equally broken. Fix: refresh on EVERY call, accepted
  or not, so the whole burst is swallowed regardless of its length.

  v2 -- a threshold calibrated on the REPEAT INTERVAL (2/repeat_rate, 80 ms
  at 25 Hz, floor 150 ms). Caught by the test bench: the FIRST repeat arrives
  `repeat_delay` after the initial press (600 ms here), well past that
  threshold -- so it slipped through, and one stray flip is enough to stop a
  recording after six tenths of a second of nothing. Fix: the threshold must
  cover `repeat_delay`, not the interval.

  v3 -- read-then-write with no locking. Deployed on utu, 2026-09-22: a real
  `Super+Shift+D` press produced "j'écoute" then "rien entendu" 0.8 ms apart
  in the daemon's own log -- two `geshtu dictee` processes, launched close
  enough together that BOTH read the jeton file before EITHER had written
  it, so both computed `recent = False` and both reached the daemon. A
  classic TOCTOU race: v1 and v2 fixed the THRESHOLD, but never protected
  the read-decide-write sequence itself from a second process running the
  same three steps concurrently. Confirmed NOT mango's `reload_config`
  (source read at the locked flake rev: it fully clears `key_bindings`
  before reparsing, does not append) and NOT a duplicate `bind=` line
  (checked both the repo and the live `/etc/mango/config.conf`: one line
  each for dictee and commande). Fix: hold an flock across read AND write,
  so a second process blocks until the first has committed its state, then
  reads the value the first one just wrote -- not the stale one from before
  either process ran.

The right shape:

    threshold = max(repeat_delay * 1.25, 2 / repeat_rate, 150 ms), capped at 1.5 s

Both values are READ FROM MANGO'S OWN CONFIG, never hardcoded -- a host with
`repeat_delay=1000` needs a wider threshold, and a fixed one would not help
it. Verified against three profiles (utu 600/25, slow 1000/5, fast 200/50):
zero stray flips, one log line per burst.
"""
from __future__ import annotations

import fcntl
import os
import pathlib
import subprocess
import time


def repete(nom: str) -> bool:
    """True if this call is an echo of mango's key-repeat, not a real press.

    `nom` names the toggle (e.g. "dictee", "commande") so several bindings
    each keep their own state file and never stomp on each other's timing.
    """
    delai_ms, cadence = 600.0, 25.0
    try:
        for ligne in pathlib.Path("/etc/mango/config.conf").read_text().splitlines():
            if ligne.startswith("repeat_delay="):
                delai_ms = float(ligne.split("=", 1)[1].strip())
            elif ligne.startswith("repeat_rate="):
                cadence = float(ligne.split("=", 1)[1].strip())
    except (OSError, ValueError):
        pass
    seuil = max(delai_ms / 1000.0 * 1.25, 2.0 / cadence if cadence > 0 else 0, 0.15)
    seuil = min(seuil, 1.5)

    base = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    jeton = base / ("geshtu-keyguard-" + nom)
    maintenant = time.time()
    dernier, etat_prec = 0.0, "a"

    # ⚠️ THE WHOLE READ-DECIDE-WRITE SEQUENCE RUNS UNDER ONE flock, not just
    # the write. Two `geshtu dictee` processes launched a few ms apart is
    # exactly the case this guard exists for -- if the second one could read
    # before the first writes, it would decide "not recent" from stale data
    # every time, no matter how tight the window. The lock makes the second
    # process wait for the first to finish deciding, then read what the
    # first one just committed.
    fd = os.open(jeton, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            morceaux = os.read(fd, 256).decode().split()
            dernier = float(morceaux[0])
            etat_prec = morceaux[1] if len(morceaux) > 1 else "a"
        except (ValueError, IndexError, UnicodeDecodeError):
            pass

        recent = (maintenant - dernier) < seuil

        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, ("%s %s" % (maintenant, "i" if recent else "a")).encode())
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    # ⚠️ ONE LOG LINE PER BURST, not one per repeat. At 25 Hz that would be 25
    # lines/second -- exactly the tool meant for diagnosing this. The `a`/`i`
    # marker says whether the previous call was accepted, so this one knows
    # whether it opens the burst.
    #
    # ⚠️ THIS IS THE ONE geshtu LOG LINE THAT DOES NOT GO THROUGH THE DAEMON.
    # It fires in the short-lived CLI process, before the socket is even
    # opened -- there is no `State.report()` to call yet, and there should not
    # be: the daemon was never involved in this decision. `journalctl -t
    # geshtu` (not `-u geshtu.service`) is where a key-repeat storm shows up.
    if recent and etat_prec == "a":
        try:
            subprocess.run(["systemd-cat", "-t", "geshtu"],
                           input=("IGNORE repetition clavier (%s, seuil %.0f ms)"
                                  % (nom, seuil * 1000)).encode(),
                           timeout=5, check=False)
        except Exception:                                # noqa: BLE001
            pass
    return recent
