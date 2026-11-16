# train-relay

**English** | [中文](README.zh-CN.md)

You write code on one machine and train on another. `train-relay` starts the long job over ssh, watches it, and
brings the results home. One job is one TOML file; one command does the rest.

```bash
./train-relay run job.toml
```

What it prints (an example):

```
2026-01-12 11:17:25 preflight ok (3 files present, 1 hashes, 1 absent)
2026-01-12 11:17:46 8 of 8 GPUs on gpu-server idle (need 8)
2026-01-12 11:17:47 started finetune_v2 on gpu-server; its log: /data/myproject/.train-relay/finetune_v2.log
2026-01-12 11:22:48 running step=20/742
...
2026-01-12 12:36:23 pull epoch_1.pt (12045462294 bytes) from gpu-server
2026-01-12 12:39:12   ok epoch_1.pt (md5 5859311b63d54b0c433980eb1a28a0a7)
...
2026-01-12 13:36:10 finetune_v2 finished
2026-01-12 13:39:02   ok epoch_2.pt (md5 9d1c0e5c2f0e4a8b7b1f3a6d5c4e2f10)
2026-01-12 13:39:03 DONE
```

It replaces the shell script that everyone rewrites for each training run: wait for the GPUs, launch with `nohup`,
`tail` the log every few minutes, `scp` the checkpoint, check that it arrived intact.

Python standard library only (3.11+), nothing to install. The remote needs bash, dd, stat and md5sum, plus nvidia-smi
if you use the GPU check.

## The six stages

1. **Preflight.** Files that must exist on the remote (optionally with a sha256) and paths that must not, so a run
   never starts with the wrong parent weights or on top of an earlier run.
2. **GPUs.** The number of idle cards you need, looked at twice 20 s apart. Fail if they are busy, or wait for them.
3. **Launch.** The command runs detached (`setsid nohup`) with a pid file and a log on the remote. Closing your
   laptop does not touch it.
4. **Watch.** At every poll: finished? an error line? still alive? how far? One log line each time.
5. **Pull.** Result files are fetched as they appear and at the end, newest N only. Each file is split into byte
   ranges that travel over parallel ssh connections, then compared with the remote's md5.
6. **Hooks.** A local command after each pulled file and at the end (done or failed): send a notification, register
   the checkpoint, start the next step.

## Usage

```bash
git clone https://github.com/heliang-pu/train-relay && cd train-relay

./train-relay run job.toml              # all six stages; run it in tmux or under nohup
./train-relay run job.toml --dry-run    # print the plan and the remote launch script; touch nothing
./train-relay run job.toml --attach     # the job is running already: only watch and pull
./train-relay status job.toml           # one look
./train-relay pull job.toml             # fetch the newest results now
./train-relay stop job.toml             # stop the remote job (its whole process group)
```

If the local `run` is interrupted (Ctrl-C, a closed terminal, a dropped VPN), the remote job keeps going;
`--attach` picks it up again. A failed ssh during a status poll is not a failure of the job; the next poll retries.

A complete job file is in [`examples/finetune.toml`](examples/finetune.toml).

## Job file

| Section | Key | Meaning | Default |
|---|---|---|---|
| top level | `name` | Job name: letters, digits, dot, dash, underscore | required |
| top level | `host` | ssh alias; `local` runs on this machine | required |
| top level | `workdir` | Absolute remote directory the command runs in | required |
| `[preflight]` | `exists` | Remote paths that must exist | none |
| `[preflight]` | `absent` | Remote paths that must not exist | none |
| `[preflight.sha256]` | `"path" = "hash"` | The file must have this sha256 | none |
| `[gpus]` | `count` | Idle GPUs needed; 0 = no check | 0 |
| `[gpus]` | `idle_mib` | A card below this memory use is idle | 2000 |
| `[gpus]` | `wait` | Wait for the GPUs instead of failing | false |
| `[launch]` | `command` | The command (may span lines) | required |
| `[launch.env]` | | Environment exported to the command | none |
| `[watch]` | `poll_s` | Seconds between looks | 300 |
| `[watch]` | `done` | Regex of the log line that means success; without it, exit code 0 | none |
| `[watch]` | `error` | Regex of a log line that means failure | none |
| `[watch]` | `progress` | Regex whose last match is printed | none |
| `[watch]` | `settle_s` | While running: how long a file must be unchanged to count as written | 60 |
| `[[pull]]` | `glob` | Absolute remote glob of result files | required |
| `[[pull]]` | `to` | Local directory | required |
| `[[pull]]` | `newest` | Fetch only the newest N by modification time; 0 = all | 2 |
| `[[pull]]` | `while_running` | Start fetching during the run | false |
| `[[sync]]` | `from`, `to` | Small files or directories copied with rsync | none |
| `[transfer]` | `parallel` | Byte ranges per file | 8 |
| `[transfer]` | `min_free_gb` | Do not pull when local free space would drop below this | 0 |
| `[hooks]` | `after_pull`, `on_done`, `on_fail` | Local commands | none |

Hooks receive `RELAY_NAME` and `RELAY_HOST`; `after_pull` also gets `RELAY_FILE` (local path) and `RELAY_REMOTE`;
`on_done` and `on_fail` also get `RELAY_STATE` and `RELAY_TEXT` (a one-line result).

## How the job's state is decided

| Verdict | Condition |
|---|---|
| failed | A log line matches `error` |
| done | A log line matches `done`; or, without `done`, the command exits with 0 |
| failed | The command exits with a non-zero code; or `done` is set and the command exits with 0 without printing it |
| failed | The process is gone and left no exit record (killed, or the machine restarted) |
| running | None of the above |

## What is left where

- Remote: `<workdir>/.train-relay/<name>.sh` (the launch script), `<name>.pid`, `<name>.log` (all output of the command).
- Local: `.train-relay.json` in each `to` directory records size, md5, origin and time of every fetched file. A file
  that was fetched and has not changed on the remote is not fetched again.
- A file in transit is called `.<name>.part`; it gets its real name only after the md5 matches, and is deleted
  otherwise, so a broken download never looks like a checkpoint.

## Notes

- With `while_running = true` and `newest = 2`, every new file is fetched as it appears, because at that moment it is
  one of the newest two. For many epochs of large files over a metered line, leave `while_running` off.
- `stop` ends the launch script's process group. Children that a framework starts in a session of their own are not
  in it.
- This tool handles one job. For a queue across several machines, see [gpuq](https://github.com/heliang-pu/gpuq).

## Tests

```bash
python3 -m pytest -q
```

All tests run on the local machine (`host = "local"`) against a fake training command that writes a log and result
files; they cover success, each kind of failure, the GPU wait, attach, stop, and a corrupted transfer.
