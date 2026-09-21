"""Talking to the inference endpoints. One place, so device placement stays a
configuration question and never leaks into a stage."""
from __future__ import annotations

import json
import pathlib
import urllib.request
import uuid


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
        raise RuntimeError("%s unreachable: %s" % (base, exc))
    out = []
    for name, info in sorted(raw.items()):
        versions = info.get("model_version_status") or []
        state = versions[0].get("state") if versions else "?"
        out.append({"name": name, "state": state})
    return out


def transcribe(engine, path: pathlib.Path) -> str:
    """One audio file in, text out.

    ⚠️ NEVER SEND `language=`. The model detects it, and forcing the wrong one
    produces fluent nonsense with total confidence. The summary language is a
    separate setting; they are not the same decision.

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
    body.extend(("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
                 "filename=\"%s\"\r\nContent-Type: audio/wav\r\n\r\n"
                 % (boundary, path.name)).encode())
    body.extend(path.read_bytes())
    body.extend(("\r\n--%s--\r\n" % boundary).encode())

    req = urllib.request.Request(
        engine.endpoint, data=bytes(body),
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=engine.timeout) as resp:
        return (json.loads(resp.read()).get("text") or "").strip()


def chat(engine, prompt: str, system: str = "", max_tokens: int = 1100,
         allow_truncated: bool = False) -> str:
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
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": engine.model,
        "messages": messages,
        "temperature": 0.6,
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
