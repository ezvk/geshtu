# Adding a model

geshtu does not load models; the model server does. This is how you hand it
one, and what bites.

## The server has to be watching

```sh
ovms --config_path /var/lib/geshtu/ovms.json \
     --file_system_poll_wait_seconds 5 \
     --cache_dir /var/lib/geshtu/cache --rest_port 8092
```

⚠️ Started with `--model_path` instead, the server serves exactly one model
and never reads a config file. Editing one then looks, from the outside,
exactly like a config being ignored — and it is, because there is none.

Then in `~/.config/geshtu/geshtu.toml`:

```toml
[repository]
ovms   = "ovms"
models = "/var/lib/geshtu/models"
config = "/var/lib/geshtu/ovms.json"
```

## Adding one

```sh
# a directory of OpenVINO IR already on disk
geshtu models add /var/lib/ovmodels/qwen3-8b-int4-cw-ov --name qwen3-8b \
    --task text_generation -- --target_device NPU --max_prompt_len 8192

# or straight from Hugging Face
geshtu models add hf:OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov \
    --name mistral-7b --task text_generation -- --target_device NPU
```

Everything after a bare `--` goes to the server untouched, so task options
travel without geshtu having to understand them.

Then point an engine at it, with no restart:

```sh
geshtu models                    # what the endpoints serve, and their state
geshtu model llm mistral-7b
```

## Three things that bite

⚠️ **`--configure` writes into the model directory.** `graph.pbtxt` lands
next to the weights, so the directory must be writable. Pointing this at a
read-only or shared store fails in a way that reads like a permission bug
rather than a design decision.

⚠️ **A generative model needs its task.** The graph that turns an HTTP
payload into generation is exactly what `--configure` builds; without it the
server starts and answers nothing useful.

⚠️ **Present is not served.** A directory of weights nobody listed in the
config is invisible to the server, and `geshtu models` asks the *server*. If
you can see a model on disk and not in that listing, it was never added.

## The accelerator has opinions

On an Intel NPU, prompt length is fixed at compile time and there is a hard
ceiling — measured by bisection at **8192 tokens**, with 9216 refused in 15 s
and a model a gigabyte smaller failing at exactly the same point. It is a
constant of the compiler, not a memory limit. Declare it:

```toml
[engines.llm]
max_prompt = 8192
```

so the pipeline chapters instead of sending something that will be refused.
