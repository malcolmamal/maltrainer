# Control API

A REST API, with Swagger, to check on and steer a training run from outside its console. It is served by the trainer's
own process, so it exists only while a run trains. It does everything the console keys do (`s`, `q`, `p`, `g` and the
preview settings of `e`) and reports the run's numbers. It is off unless the config asks for it.

It covers every trainer that reads the console keys: the MOSS-Image finetune and LoRA, the Krea 2 finetune, and the
other LoRA trainers built on `StepLoop` (`maltrainer/training/loop.py`). What a trainer lacks is refused, not faked:
only the MOSS-Image finetune has the gallery, and only the MOSS-Image trainers have preview settings.

## Switch it on

```yaml
control_api:
  enabled: true      # off unless set
  host: 127.0.0.1    # loopback by default; any other address needs a token
  port: 8765         # 0 = a free port (the log line says which)
  token: ""          # when set, every call must send "Authorization: Bearer <token>"
```

The run logs `Control API: http://127.0.0.1:8765 (Swagger UI at /docs)` before its first step. Open
`http://127.0.0.1:8765/docs` for Swagger UI, which lists every call and lets you try it; the schema is at
`/openapi.json`. Swagger UI loads its scripts from a CDN, so `/docs` needs internet access; the API itself does not.

A mistake in the section (an unknown key, a bad port, a non-loopback host without a token) stops the run at its start
with the reason. A port that is already in use does not: the run logs an error and trains without the API, because the
run comes first.

## What it can do

| Call | Console key | What it does |
|---|---|---|
| `GET /status` | | State, step, total, percent, epoch, learning rate, ETA and the monitor's window |
| `POST /commands/save` | `s` | Save a checkpoint now; training continues |
| `POST /commands/save-and-stop` | `q` | Save, then stop (only once the save worked). Resume with `--resume-from <name>_state` |
| `POST /commands/pause` | `p` | Wait at the next step boundary |
| `POST /commands/resume` | `p` again | Carry on |
| `POST /commands/gallery` | `g` | Render the gallery (MOSS-Image finetune; training waits) |
| `GET /preview-settings` | `e` | The interval, the raw/EMA rows and each prompt with its slot and on/off |
| `PATCH /preview-settings` | `e` | Change the interval, rows or which prompts render |
| `GET /commands`, `GET /commands/{id}` | | Recent commands and where each stands |

There is no call that stops without saving, and none that starts a run: a stopped run is resumed from its `_state`
folder as before, from the command line.

### Commands are asynchronous

The loop acts on a command at its next step boundary, the same place it reads the console keys. A command call answers
`202` at once with an id:

```
$ curl -X POST http://127.0.0.1:8765/commands/save-and-stop
{"id": 3, "command": "save-and-stop", "state": "pending", "detail": "", "submitted_at": 1761040000.1, "settled_at": null}
$ curl http://127.0.0.1:8765/commands/3
{"id": 3, "command": "save-and-stop", "state": "done", "detail": "saved", ...}
```

`state` is `pending`, `done` or `failed`. A `save-and-stop` whose save failed (a full disk) is `failed` and training
continues, exactly as with `q`. While the run is paused the commands act at once. A call is refused with `409` when the
run is stopping or finished, when the trainer lacks the feature, when a `save-and-stop` already waits, or when 50
commands already wait for the next step. A command still waiting when the run ends fails with that reason. A `pause`
while paused, a `resume` while running, or a `pause` followed by a `resume` in one step change nothing and settle as
done.

The API and the console keys share one code path. An API command becomes the key it stands for, so `s`, `q`, `p` and `g`
behave identically from either, and both work together.

### Status

`GET /status` reports `state`: `running`, `paused`, `saving`, `gallery`, `stopping` or `finished`. The numbers are read
once per optimizer step, so `published_seconds_ago` tells how fresh they are: during a long preview round or a save
they are as old as that round. `window` is what the trainer logs as its `[monitor]` line, since the last one: the mean
loss, the loss per noise band, the prior's mean and the seconds per step (without previews and pauses). `eta_seconds`
is that rate times the steps left, an estimate.

### Preview settings

`PATCH /preview-settings` takes any of `every`, `rows` and `prompts`, and only what you send changes:

```
curl -X PATCH http://127.0.0.1:8765/preview-settings -H "Content-Type: application/json" \
  -d '{"every": 250, "rows": {"raw": false}, "prompts": [{"slot": 0, "enabled": false}, {"slot": 1, "enabled": true}]}'
```

A prompt is named by its `slot`, the stable number it keeps for its seed and sample file name (see `prompts.txt`).
Prompts are defined in the config, as for the `e` menu; the API switches them on and off, it does not add text. Like the
menu, a change lasts for the run and the config file is untouched. A patch is applied whole or not at all, and a slot
that no longer exists fails the command. `GET /preview-settings` follows changes made any way (this API, the `e` menu,
saving the config file, which applies its `sample:` section) at the next step.

## Security

The API can stop your run, so it is closed by default and narrow when open:

- It binds to `127.0.0.1` unless you set `host`, and a non-loopback `host` is refused without a `token`.
- With a `token`, every call needs `Authorization: Bearer <token>` (Swagger UI's Authorize button sends it).
- A request that a browser sends from another origin is refused with `403`, and, without a token, so is one whose
  `Host` is not the configured address. That keeps a web page you happen to visit from posting to `localhost`, and
  defeats DNS rebinding. With a token the API may be reached under any name (a `0.0.0.0` bind, a LAN address), since a
  page on another site cannot send the token. `curl` and scripts send no `Origin` and pass.
- The token is in the config file in plain text, like the rest of it; keep the file as private as you keep the run.

## Design notes

- **One run, one API.** Each training process serves its own API on its own port; there is no standalone service.
- **It never ends the run.** A failure reading the stats or applying a preview edit is logged and the training goes on.
  The API stops about a second after the loop ends, before the final save, and its state then reads `finished`.
- **The training thread owns the trainer.** The API thread only queues commands and reads a snapshot the training thread
  publishes at each step boundary. Nothing in the trainer is touched from another thread, so a save cannot race a step.
- **Where it lives.** `maltrainer/control/`: `hub.py` (the queue and snapshots, no web code), `keys.py`
  (`ControlledKeys`, the key handling extended), `app.py` (FastAPI), `server.py` (uvicorn in a daemon thread),
  `launch.py` (what a trainer calls). The package sits above `utils`; see the table in
  [Architecture](architecture.md#61-layers-and-dependency-rules).
