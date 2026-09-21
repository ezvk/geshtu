"""ffmpeg wrappers, and the three traps they exist to avoid.

Every function here was written after something silently produced plausible
garbage. Read the warnings before simplifying any of them.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

BYTES_PER_SEC = 32000          # 16 kHz, mono, signed 16-bit


def duration(path: pathlib.Path) -> float:
    """Duration in seconds, TOLERATING A FILE STILL BEING WRITTEN.

    A recorder only writes the size into the RIFF header when it CLOSES the
    file. While capture is running ffprobe returns the literal string "N/A",
    and float() on it raises. That crash lands in the status command, which
    means every UI polling it goes blank -- while the recording is in fact
    working perfectly. A failure in the display is the worst place to have
    one: it makes a healthy capture look dead.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        pass
    try:
        return max(path.stat().st_size - 44, 0) / BYTES_PER_SEC
    except OSError:
        return 0.0


def level_db(path: pathlib.Path) -> float | None:
    """Mean volume. Note that ffmpeg prints this at INFO level: passing
    -loglevel quiet silently removes the very line being parsed."""
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None


def mix(tracks: list[list[pathlib.Path]], dst: pathlib.Path,
        rate: int = 16000) -> bool:
    """Concatenate the segments of each track, then MIX the tracks together.

    ⚠️ TWO DIFFERENT OPERATIONS, AND CONFUSING THEM RUINS EVERYTHING.
    Segments of one track follow each other in time, so they concatenate.
    The tracks themselves are simultaneous, so they mix. Gluing all four files
    end to end doubles the duration, halves the level and invalidates every
    timestamp -- and the result still looks like a valid recording.

    ⚠️ The caller must pass EVERY track. An earlier version iterated over a
    hardcoded list of track kinds and silently dropped a later addition: the
    audio was captured at -20.3 dB, written to disk, and then discarded at mix
    time. The per-track levels reported at stop time read the SEGMENTS, so they
    happily announced a track the rest of the chain ignored. A log that is
    truthful about one step says nothing about the next.

    ⚠️ normalize=0 is required. Without it amix divides by the number of
    inputs, so mixing a silent track into a spoken one halves the speech.
    """
    joined: list[pathlib.Path] = []
    for i, segments in enumerate(tracks):
        if not segments:
            continue
        if len(segments) == 1:
            joined.append(segments[0])
            continue
        listing = dst.parent / ("concat-%d.txt" % i)
        listing.write_text("".join("file '%s'\n" % s for s in segments))
        out = dst.parent / ("track-%d.wav" % i)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-ac", "1", "-ar", str(rate), str(out)], check=True)
        joined.append(out)

    if not joined:
        return False
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for j in joined:
        cmd += ["-i", str(j)]
    if len(joined) == 1:
        cmd += ["-ac", "1", "-ar", str(rate), str(dst)]
    else:
        cmd += ["-filter_complex",
                "amix=inputs=%d:duration=longest:normalize=0" % len(joined),
                "-ac", "1", "-ar", str(rate), str(dst)]
    subprocess.run(cmd, check=True)
    return True


def silence_cuts(path: pathlib.Path, threshold: str = "-30dB",
                 minimum: float = 0.4) -> list[float]:
    """Where the speech pauses, in seconds.

    ⚠️ Slicing at a fixed interval cuts through the middle of a word and the
    ASR returns two confident halves of nothing. Cut on silence instead.
    """
    r = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af",
         "silencedetect=noise=%s:d=%s" % (threshold, minimum),
         "-f", "null", "-"], capture_output=True, text=True)
    return [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", r.stderr)]


def slice_out(src: pathlib.Path, dst: pathlib.Path, start: float,
              end: float, rate: int = 16000) -> pathlib.Path:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", "%.3f" % start, "-to", "%.3f" % end, "-i", str(src),
                    "-ac", "1", "-ar", str(rate), str(dst)], check=True)
    return dst


def is_silent(path: pathlib.Path, rms_threshold: float = 120.0) -> bool:
    """Guard against ASR hallucination on an empty recording.

    ⚠️ FIND THE `data` CHUNK, never assume it starts at byte 44. Different
    writers emit different header lengths; reading 13 header bytes as audio
    raised a measured RMS from 0.7 to 335 and let a silent file pass. That bug
    sat dormant for months because the wrong answer was still under the
    threshold, by luck.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return True
    at = blob.find(b"data")
    body = blob[at + 8:] if at >= 0 else blob[44:]
    if len(body) < 2:
        return True
    import array
    pcm = array.array("h")
    pcm.frombytes(body[:len(body) - (len(body) % 2)])
    if not pcm:
        return True
    rms = (sum(float(v) * v for v in pcm) / len(pcm)) ** 0.5
    return rms < rms_threshold
