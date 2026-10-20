"""Talking to the inference endpoints. One place, so device placement stays a
configuration question and never leaks into a stage."""
from __future__ import annotations

import json
import pathlib
import urllib.request
import re
import uuid


def endpoint_for(saisie: str, configured: str) -> str:
    """The address typed in the window, made into a full endpoint.

    What one types is a SERVER -- « localhost:1337 », « http://utu:8080/v1 »
    -- while an engine needs the full route it posts to. The route is taken
    from the configured endpoint, so an llm engine stays on
    /v1/chat/completions and an embed engine on /v1/embeddings whatever the
    server. A full route typed in is kept as is. Empty = back to the config.
    """
    url = saisie.strip()
    if not url:
        return configured
    if "://" not in url:
        url = "http://" + url
    if re.search(r"/v[13]/.+", url):
        return url
    base = re.sub(r"/v[13]$", "", url.rstrip("/"))
    m = re.search(r"/v[13]/.*$", configured)
    return base + (m.group(0) if m else "/v1/chat/completions")


def available(engine) -> list[dict]:
    """The models this endpoint currently serves, with their state.

    ⚠️ THE MANAGEMENT API STAYS ON /v1 even on a server whose inference routes
    are /v3. Asking /v3/config returns 404 and reads exactly like a server
    that is down, which is a long way to travel for a wrong conclusion.

    ⚠️ AND "LOADED" IS NOT "USABLE": a model server reports a model as present
    while it is still compiling for an accelerator. The state is returned
    alongside the name rather than filtered out, so the caller can say so.
    """
    base = engine.endpoint.split("/v1/")[0].split("/v3/")[0].rstrip("/")
    try:
        with urllib.request.urlopen(base + "/v1/config", timeout=10) as resp:
            raw = json.loads(resp.read())
    except Exception as exc:                            # noqa: BLE001
        # ⚠️ /v1/config IS OVMS ONLY. Any other OpenAI-compatible server
        # (LiteLLM, llama-server, vLLM) lists on /v1/models instead, and has no
        # per-model state to report: "listed" says exactly that -- the name is
        # routable, nothing is claimed about whether it is loaded. Without this,
        # pointing an engine at LiteLLM showed "unreachable" for a server that
        # answers every request.
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=10) as resp:
                listed = json.loads(resp.read()).get("data") or []
        except Exception:                               # noqa: BLE001
            raise RuntimeError("%s unreachable: %s" % (base, exc))
        return [{"name": m["id"], "state": "listed"}
                for m in sorted(listed, key=lambda m: m["id"])]
    out = []
    for name, info in sorted(raw.items()):
        versions = info.get("model_version_status") or []
        state = versions[0].get("state") if versions else "?"
        out.append({"name": name, "state": state})
    return out


def transcribe(engine, path: pathlib.Path, language: str | None = None) -> str:
    """One audio file in, text out.

    ⚠️ THE LANGUAGE IS SENT. This docstring used to say "NEVER SEND
    `language=`, the model detects it". Measured on OVMS 2026.4.0 (utu, NPU,
    2026-09-27): it does NOT detect per request -- each request reuses the
    LAST language. JFK came out « Et donc mes amis américains… » right after a
    French clip, and French came out « Hello, I'm the local voice… » right
    after JFK. `language=` empty resets nothing, `language=auto` is an error.
    With an explicit language the text was right every time. The spoken
    language is therefore a setting (State.langue_entree()), SEPARATE from the
    summary language -- ezvk: « je préfère choisir séparément le langage
    d'entrée, ça va pas seulement servir aux réunions ».

    ⚠️ `response_format` is ignored by OpenVINO Model Server: verbose_json,
    srt and vtt all return a bare {"text": ...}. There are no timestamps to
    recover, which is why the caller slices and remembers where it cut.
    """
    boundary = uuid.uuid4().hex
    body = bytearray()

    def field(name, value):
        body.extend(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                     % (boundary, name, value)).encode())

    field("model", engine.model)
    if language:
        field("language", language)
    body.extend(("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
                 "filename=\"%s\"\r\nContent-Type: audio/wav\r\n\r\n"
                 % (boundary, path.name)).encode())
    body.extend(path.read_bytes())
    body.extend(("\r\n--%s--\r\n" % boundary).encode())

    req = urllib.request.Request(
        engine.endpoint, data=bytes(body),
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=engine.timeout) as resp:
        return nettoie((json.loads(resp.read()).get("text") or "").strip())


def chat(engine, prompt: str, system: str = "", max_tokens: int = 1100,
         allow_truncated: bool = False,
         history: list[tuple[str, str]] | None = None,
         temperature: float = 0.6) -> str:
    """One prompt in, one answer out.

    ⚠️ temperature 0.6, never 0: at zero, Qwen3 degenerates into loops.
    ⚠️ enable_thinking false, or thousands of tokens are spent reasoning
       before a single word of the answer appears.
    ⚠️ Put the transcript FIRST and the instruction LAST. With prefix caching,
       a second question about the same meeting then reuses the prefill and is
       nearly free.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    # Earlier exchanges, as real turns: the model sees what it already said.
    for question, answer in history or []:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": engine.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        engine.endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=engine.timeout) as resp:
        out = json.loads(resp.read())
    choice = (out.get("choices") or [{}])[0]
    # ⚠️ `length` means the answer was CUT. Concluding anything from a
    # truncated answer is how invented "decisions" get into a report -- so
    # this raises by default, and the caller decides whether a cut answer is
    # better than none. It is NOT the caller's job to guess: the flag makes
    # the choice visible at the call site.
    texte = (choice.get("message") or {}).get("content", "").strip()
    if choice.get("finish_reason") == "length":
        if not allow_truncated:
            raise RuntimeError("answer truncated (max_tokens=%d)" % max_tokens)
        return texte
    return texte


def embed(engine, texts: list[str]) -> list[list[float]]:
    payload = {"model": engine.model, "input": texts}
    req = urllib.request.Request(
        engine.endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=engine.timeout) as resp:
        out = json.loads(resp.read())
    return [d["embedding"] for d in out.get("data", [])]


def speak(engine, text: str, voice: str, language: str) -> bytes:
    """Text in, WAV bytes out (OVMS /v3/audio/speech, OpenAI-shaped).

    ⚠️ `language` IS NOT OPTIONAL IN PRACTICE: OVMS defaults to en-us, and
    French phonemised as English is broken speech, not an accent.
    """
    payload = {"model": engine.model, "voice": voice, "language": language,
               "input": text}
    req = urllib.request.Request(
        engine.endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=engine.timeout) as resp:
        return resp.read()


# -- Whisper's repetition loops -------------------------------------------
#
# ⚠️ ON NON-SPEECH WHISPER LOOPS, AND THE LOOP IS NOT SPEECH. Seen live on
# 2026-10-20 (whisper-large-v3-turbo int8, a Teams call): keyboard typing over
# 75 seconds came out as « ça tatatata ta ta ta ta ta … » a few hundred times;
# silence or music gives « Oooooo… ». ezvk: « il retranscrit n'importe quoi ».
# A word or group of up to four words repeated more than 3 times in a row is
# kept once; a letter pattern repeated 4 times or more inside a word is
# kept once; what is mostly loop is dropped. Applied to every transcription
# (meeting, live view, dictation): an honest repetition is never 4+ in a row.

_MOT = re.compile(r"[\w'’-]+", re.UNICODE)
_RUN = re.compile(r"([^\W\d_]{1,3}?)\1{3,}", re.UNICODE)   # letters only: 1000000 stays


def _cle(mot: str) -> str:
    return "".join(_MOT.findall(mot.lower()))


def nettoie(texte: str) -> str:
    """Collapse Whisper repetition loops; '' when the text is mostly loop."""
    if not texte:
        return texte
    mots = [_RUN.sub(r"\1", m) for m in texte.split()]
    avant = len(mots)
    sortie: list[str] = []
    i = 0
    while i < len(mots):
        garde = False
        for n in (1, 2, 3, 4):
            motif = [_cle(m) for m in mots[i:i + n]]
            if len(motif) < n or not all(motif):
                continue
            k = 1
            while [_cle(m) for m in mots[i + k * n:i + (k + 1) * n]] == motif:
                k += 1
            if k > 3:                       # 4 occurrences or more: keep one
                sortie.extend(mots[i:i + n])
                i += k * n
                garde = True
                break
        if not garde:
            sortie.append(mots[i])
            i += 1
    if avant >= 12 and len(sortie) <= avant * 0.2 and len(sortie) <= 6:
        return ""                           # the phrase WAS the loop
    return " ".join(sortie)
