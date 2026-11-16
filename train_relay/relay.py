"""train-relay: start a long job on another machine over ssh, watch it, and bring its results home.

One TOML file describes the job. ``train-relay run job.toml`` then does, in order:

  1. preflight  files that must exist on the remote (optionally with a sha256), paths that must not
  2. gpus       enough idle GPUs (checked twice), or wait for them
  3. launch     the command, detached (setsid nohup), with a pid file and a log on the remote
  4. watch      every poll: finished? an error line? still alive? how far? -- one line of log each time
  5. pull       result files as they appear and at the end: parallel byte ranges over ssh, md5 checked, newest N only
  6. hooks      a local command after each pulled file, and at the end (done / failed)

Python standard library only (3.11+). The remote needs bash, dd, stat, md5sum (and nvidia-smi for the GPU check).
host = "local" runs everything on this machine, which is also how the tests work.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

VERSION = "0.1.0"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "ConnectTimeout=30",
       "-o", "ServerAliveInterval=30"]  # fmt: skip
GPU_QUERY = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
EXIT_MARK = "[train-relay] exit"


class RelayError(Exception):
    """The job cannot go on; the CLI prints it and exits with 1."""


def log(text: str) -> None:
    print(f"{datetime.datetime.now():%F %T} {text}", flush=True)


@dataclasses.dataclass
class Pull:
    glob: str
    to: Path
    newest: int = 2
    while_running: bool = False


@dataclasses.dataclass
class Job:
    name: str
    host: str
    workdir: str
    command: str
    env: dict
    exists: list
    absent: list
    sha256: dict
    gpu_count: int
    gpu_idle_mib: int
    gpu_wait: bool
    gpu_recheck_s: float
    gpu_query: str
    poll_s: float
    done: str
    error: str
    progress: str
    settle_s: float
    pulls: list
    sync: list
    parallel: int
    min_range_mb: float
    stagger_s: float
    min_free_gb: float
    hooks: dict
    source: Path

    @property
    def state_dir(self) -> str:
        return f"{self.workdir.rstrip('/')}/.train-relay"

    def remote(self, kind: str) -> str:
        return f"{self.state_dir}/{self.name}.{kind}"


def load(path: Path | str) -> Job:
    path = Path(path)
    d = tomllib.loads(path.read_text())
    known = {"name", "host", "workdir", "preflight", "gpus", "launch", "watch", "pull", "sync", "transfer", "hooks"}
    unknown = set(d) - known
    if unknown:
        raise RelayError(f"{path}: unknown keys {sorted(unknown)}")
    for key in ("name", "host", "workdir"):
        if not d.get(key):
            raise RelayError(f"{path}: {key} is missing")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", d["name"]):
        raise RelayError(f"{path}: name may hold letters, digits, dot, dash and underscore only")
    if not d["workdir"].startswith("/"):
        raise RelayError(f"{path}: workdir must be an absolute path on the remote")
    pre, gpus, launch, watch = d.get("preflight", {}), d.get("gpus", {}), d.get("launch", {}), d.get("watch", {})
    transfer = d.get("transfer", {})
    if not launch.get("command"):
        raise RelayError(f"{path}: [launch] command is missing")
    pulls = []
    for item in d.get("pull", []):
        if not item.get("glob", "").startswith("/"):
            raise RelayError(f"{path}: [[pull]] glob must be an absolute remote path pattern")
        pulls.append(Pull(item["glob"], Path(os.path.expanduser(item["to"])), int(item.get("newest", 2)),
                          bool(item.get("while_running", False))))  # fmt: skip
    sync = [{"from": list(s["from"]), "to": Path(os.path.expanduser(s["to"]))} for s in d.get("sync", [])]
    return Job(
        name=d["name"], host=d["host"], workdir=d["workdir"], command=launch["command"],
        env={str(k): str(v) for k, v in launch.get("env", {}).items()},
        exists=list(pre.get("exists", [])), absent=list(pre.get("absent", [])), sha256=dict(pre.get("sha256", {})),
        gpu_count=int(gpus.get("count", 0)), gpu_idle_mib=int(gpus.get("idle_mib", 2000)),
        gpu_wait=bool(gpus.get("wait", False)), gpu_recheck_s=float(gpus.get("recheck_s", 20)),
        gpu_query=gpus.get("query_command", GPU_QUERY),
        poll_s=float(watch.get("poll_s", 300)), done=watch.get("done", ""), error=watch.get("error", ""),
        progress=watch.get("progress", ""), settle_s=float(watch.get("settle_s", 60)),
        pulls=pulls, sync=sync, parallel=int(transfer.get("parallel", 8)),
        min_range_mb=float(transfer.get("min_range_mb", 64)), stagger_s=float(transfer.get("stagger_s", 2)),
        min_free_gb=float(transfer.get("min_free_gb", 0)), hooks=dict(d.get("hooks", {})), source=path.resolve(),
    )  # fmt: skip


# ---------------------------------------------------------------- remote execution


def argv(host: str, script: str) -> list[str]:
    return ["bash", "-c", script] if host == "local" else [*SSH, "-n", host, script]


def remote(host: str, script: str, check: bool = True, stdin: str | None = None, timeout: float = 600) -> str:
    """Run a shell script on the host; its stdout. ``stdin``: text for the script's standard input."""
    command = argv(host, script)
    if stdin is not None and host != "local":
        command.remove("-n")
    try:
        done = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RelayError(f"{host}: no answer in {timeout:g} s") from error
    if check and done.returncode != 0:
        raise RelayError(f"{host}: exit {done.returncode}: {done.stderr.strip()[-300:] or script[:120]}")
    return done.stdout


def q(text: str) -> str:
    return shlex.quote(str(text))


# ---------------------------------------------------------------- stages


def preflight(job: Job) -> None:
    lines = [f"test -e {q(p)} || echo MISSING {q(p)}" for p in [*job.exists, *job.sha256]]
    lines += [f"test -e {q(p)} && echo PRESENT {q(p)}" for p in job.absent]
    lines += [f"test -d {q(job.workdir)} || echo MISSING {q(job.workdir)}"]
    out = remote(job.host, "\n".join(lines) + "\ntrue").strip()
    if out:
        raise RelayError("preflight: " + "; ".join(
            line.replace("MISSING", "missing on the remote:").replace("PRESENT", "exists already:")
            for line in out.splitlines()))  # fmt: skip
    for path, want in job.sha256.items():
        got = remote(job.host, f"sha256sum {q(path)} | cut -c1-64", timeout=3600).strip()
        if got != want.lower():
            raise RelayError(f"preflight: sha256 of {path} is {got[:16]}..., expected {want[:16]}...")
    log(
        f"preflight ok ({len(job.exists) + len(job.sha256)} files present, {len(job.sha256)} hashes, {len(job.absent)} absent)"
    )


def idle_gpus(job: Job) -> tuple[int, int]:
    """(idle, total) on the host."""
    used = [int(float(x)) for x in remote(job.host, job.gpu_query).split() if x.strip()]
    return sum(1 for mib in used if mib < job.gpu_idle_mib), len(used)


def gpus(job: Job) -> None:
    if not job.gpu_count:
        return
    waited = False
    while True:
        first = idle_gpus(job)
        if first[0] >= job.gpu_count:
            time.sleep(job.gpu_recheck_s)  # a job that was loading shows up the second time
            second = idle_gpus(job)
            if second[0] >= job.gpu_count:
                log(f"{second[0]} of {second[1]} GPUs on {job.host} idle (need {job.gpu_count})")
                return
            first = second
        if not job.gpu_wait:
            raise RelayError(
                f"{job.host}: {first[0]} of {first[1]} GPUs idle, the job needs {job.gpu_count} ([gpus] wait = true waits for them)"
            )
        if not waited:
            log(f"waiting for {job.gpu_count} idle GPUs on {job.host} ({first[0]} of {first[1]} idle now)")
            waited = True
        time.sleep(job.poll_s)


def launch_script(job: Job) -> str:
    exports = "".join(f"export {k}={q(v)}\n" for k, v in job.env.items())
    return (
        "#!/bin/bash\n"
        f"# written by train-relay {VERSION} from {job.source} on {datetime.datetime.now():%F %T}\n"
        f"echo $$ > {q(job.remote('pid'))}\n"
        f"cd {q(job.workdir)} || exit 97\n"
        f"{exports}"
        f"(\n{job.command}\n)\n"  # a subshell: an "exit" inside the command must not skip the exit line below
        "code=$?\n"
        f'echo "{EXIT_MARK} $code"\n'
        "exit $code\n"
    )


def is_running(job: Job) -> bool:
    out = remote(job.host, f"kill -0 $(cat {q(job.remote('pid'))} 2>/dev/null) 2>/dev/null && echo 1 || echo 0")
    return out.strip() == "1"


def launch(job: Job) -> None:
    remote(job.host, f"mkdir -p {q(job.state_dir)}")
    if is_running(job):
        raise RelayError(f"{job.name} is running on {job.host} already (--attach watches it instead of starting it)")
    remote(job.host, f"cat > {q(job.remote('sh'))}", stdin=launch_script(job))
    remote(
        job.host,
        f"rm -f {q(job.remote('pid'))}; setsid nohup bash {q(job.remote('sh'))} > {q(job.remote('log'))} 2>&1 < /dev/null &",
    )
    log(f"started {job.name} on {job.host}; its log: {job.remote('log')}")


@dataclasses.dataclass
class Status:
    exit_code: int | None
    alive: bool
    errors: int
    done: int
    progress: str
    error_line: str = ""


def status(job: Job) -> Status:
    logf = q(job.remote("log"))
    count = lambda pattern: f"grep -a -c -E {q(pattern)} {logf} 2>/dev/null || true" if pattern else "echo 0"  # noqa: E731
    last = f"grep -a -o -E {q(job.progress)} {logf} 2>/dev/null | tail -1" if job.progress else "true"
    script = "\n".join([
        f"echo \"exit=$(grep -a -F {q(EXIT_MARK)} {logf} 2>/dev/null | tail -1 | awk '{{print $NF}}')\"",
        f"echo \"alive=$(kill -0 $(cat {q(job.remote('pid'))} 2>/dev/null) 2>/dev/null && echo 1 || echo 0)\"",
        f"echo \"errors=$({count(job.error)})\"",
        f"echo \"done=$({count(job.done)})\"",
        f"echo \"progress=$({last})\"",
        (f"echo \"line=$(grep -a -E {q(job.error)} {logf} 2>/dev/null | head -1 | cut -c1-200)\"" if job.error else "true"),
    ])  # fmt: skip
    fields = dict(line.split("=", 1) for line in remote(job.host, script).splitlines() if "=" in line)
    number = lambda key: int(fields.get(key, "0").strip() or 0)  # noqa: E731
    code = fields.get("exit", "").strip()
    return Status(int(code) if code.lstrip("-").isdigit() else None, fields.get("alive", "0").strip() == "1",
                  number("errors"), number("done"), fields.get("progress", "").strip(), fields.get("line", "").strip())  # fmt: skip


def verdict(job: Job, s: Status) -> str:
    """ "running", "done", or "failed: <why>"."""
    if s.errors:
        return f"failed: an error line in the log: {s.error_line}"
    if job.done and s.done:
        return "done"
    if s.exit_code is not None:
        if s.exit_code != 0:
            return f"failed: the command exited with {s.exit_code}"
        return "failed: the command exited with 0 but the done line never appeared" if job.done else "done"
    if not s.alive:
        return "failed: the launcher is gone and left no exit line (killed, or the machine restarted)"
    return "running"


# ---------------------------------------------------------------- pulling


def listing(job: Job, pattern: str) -> list[dict]:
    """The remote files that match: name, size, mtime, and the remote's clock."""
    script = (
        f'now=$(date +%s); for f in {pattern}; do [ -f "$f" ] && echo "$now $(stat -c "%Y %s" "$f") $f"; done; true'
    )
    files = []
    for line in remote(job.host, script).splitlines():
        now, mtime, size, name = line.split(" ", 3)
        files.append({"path": name, "size": int(size), "mtime": int(mtime), "age": int(now) - int(mtime)})
    return sorted(files, key=lambda f: (f["mtime"], f["path"]))


def pull_range(job: Job, path: str, tmp: Path, offset: int, length: int) -> bool:
    script = f"dd if={q(path)} iflag=skip_bytes,count_bytes skip={offset} count={length} bs=4M status=none"
    for attempt in range(1, 6):
        process = subprocess.Popen(argv(job.host, script), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        got = 0
        with open(tmp, "r+b") as out:
            out.seek(offset)
            while chunk := process.stdout.read(4 << 20):
                out.write(chunk)
                got += len(chunk)
        if process.wait() == 0 and got == length:
            return True
        log(f"  range {offset}+{length} try {attempt}: exit {process.returncode}, {got} bytes")
        time.sleep(min(5 * attempt, 20) if job.host != "local" else 0)
    return False


def local_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324  (an integrity check against the remote's md5sum, not security)
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pulled(to: Path) -> dict:
    record = to / ".train-relay.json"
    return json.loads(record.read_text()) if record.exists() else {}


def pull_file(job: Job, item: dict, to: Path) -> bool:
    """One remote file into ``to``; True if it is there afterwards with the remote's md5."""
    name = Path(item["path"]).name
    record = pulled(to)
    known = record.get(name)
    if known and known["size"] == item["size"] and known["mtime"] == item["mtime"] and (to / name).exists():
        return True
    to.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(to).free / 2**30
    if free_gb - item["size"] / 2**30 < job.min_free_gb:
        log(f"  {name}: only {free_gb:.0f} GB free in {to}, not pulled")
        return False
    tmp = to / f".{name}.part"
    log(f"pull {name} ({item['size']} bytes) from {job.host}")
    with ThreadPoolExecutor(job.parallel + 1) as pool:
        remote_md5 = pool.submit(remote, job.host, f"md5sum {q(item['path'])} | cut -c1-32", True, None, 7200)
        with open(tmp, "wb") as f:
            f.truncate(item["size"])
        ranges = max(1, min(job.parallel, -(-item["size"] // max(1, int(job.min_range_mb * 2**20)))))
        step = -(-item["size"] // ranges) if item["size"] else 0
        jobs = []
        for i in range(ranges if item["size"] else 0):
            offset = i * step
            jobs.append(pool.submit(pull_range, job, item["path"], tmp, offset, min(step, item["size"] - offset)))
            if i + 1 < ranges and job.host != "local":
                time.sleep(job.stagger_s)  # many ssh connections at once get refused
        ok = all(j.result() for j in jobs)
        try:
            want = remote_md5.result().strip()
        except RelayError as error:
            log(f"  {name}: no md5 from the remote ({error})")
            want = ""
    got = local_md5(tmp) if ok else ""
    if not ok or not want or got != want:
        log(f"  {name}: failed" + (f" (md5 {got[:8]} here, {want[:8]} there)" if ok else " (a range failed)"))
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(to / name)
    record[name] = {"size": item["size"], "mtime": item["mtime"], "md5": got, "from": f"{job.host}:{item['path']}",
                    "time": f"{datetime.datetime.now():%F %T}"}  # fmt: skip
    (to / ".train-relay.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"  ok {name} (md5 {got})")
    hook(job, "after_pull", RELAY_FILE=str(to / name), RELAY_REMOTE=item["path"])
    return True


def sync(job: Job) -> bool:
    ok = True
    for entry in job.sync:
        entry["to"].mkdir(parents=True, exist_ok=True)
        sources = [p if job.host == "local" else f"{job.host}:{p}" for p in entry["from"]]
        done = subprocess.run(["rsync", "-a", *sources, f"{entry['to']}/"], capture_output=True, text=True)
        if done.returncode != 0:
            log(f"  sync to {entry['to']} failed: {done.stderr.strip()[-200:]}")
            ok = False
    return ok


def pull_ready(job: Job, seen: dict, finished: bool) -> bool:
    """Pull what is ready. While the job runs: files of a [[pull]] with while_running, unchanged since the last poll
    and older than settle_s. At the end: the newest N of every [[pull]]. True if nothing is left to pull."""
    complete = True
    synced = False
    for spec in job.pulls:
        files = listing(job, spec.glob)
        newest = files[-spec.newest :] if spec.newest > 0 else files
        if not finished and not spec.while_running:
            continue
        for item in newest:
            key = (item["path"], item["size"], item["mtime"])
            settled = finished or (seen.get(item["path"]) == key and item["age"] >= job.settle_s)
            seen[item["path"]] = key
            if not settled:
                complete = False
                continue
            if not synced and job.sync:
                sync(job)
                synced = True
            complete = pull_file(job, item, spec.to) and complete
        if finished and not files:
            log(f"  nothing matches {spec.glob}")
    if finished and job.sync and not synced:
        complete = sync(job) and complete
    return complete


def hook(job: Job, name: str, **env: str) -> None:
    command = job.hooks.get(name)
    if not command:
        return
    done = subprocess.run(["bash", "-c", command], env={**os.environ, "RELAY_NAME": job.name, "RELAY_HOST": job.host, **env},
                          capture_output=True, text=True)  # fmt: skip
    if done.returncode != 0:
        log(f"  hook {name} exited with {done.returncode}: {done.stderr.strip()[-200:]}")


# ---------------------------------------------------------------- commands


def watch(job: Job) -> str:
    """Poll until the job ends; pull on the way. The verdict."""
    seen: dict = {}
    misses = 0
    while True:
        try:
            s = status(job)
            state = verdict(job, s)
            misses = 0
        except RelayError as error:
            misses += 1
            log(f"no status ({error}); try {misses}")
            time.sleep(job.poll_s)
            continue
        if state != "running":
            return state
        log(f"running {s.progress or '(no progress line yet)'}")
        try:
            pull_ready(job, seen, finished=False)
        except RelayError as error:
            log(f"  pull put off: {error}")
        time.sleep(job.poll_s)


def finish(job: Job, state: str) -> int:
    if state != "done":
        log(f"FAILED {job.name}: {state[len('failed: ') :]}")
        tail = remote(job.host, f"tail -n 5 {q(job.remote('log'))} | cut -c1-200", check=False).rstrip()
        if tail:
            print(tail, flush=True)
        hook(job, "on_fail", RELAY_STATE="failed", RELAY_TEXT=f"{job.name} on {job.host}: {state}")
        return 1
    log(f"{job.name} finished")
    left = True
    for attempt in range(3):
        if pull_ready(job, {}, finished=True):
            left = False
            break
        time.sleep(min(job.poll_s, 120))
    if left:
        log(
            f"FAILED {job.name}: the job finished, but not all results were pulled (train-relay pull {job.source.name} tries again)"
        )
        hook(job, "on_fail", RELAY_STATE="pull_failed", RELAY_TEXT=f"{job.name}: finished, results not all pulled")
        return 1
    hook(job, "on_done", RELAY_STATE="done", RELAY_TEXT=f"{job.name} on {job.host}: done")
    log("DONE")
    return 0


def describe(job: Job) -> None:
    print(f"job {job.name} on {job.host}, workdir {job.workdir}")
    print(f"preflight: {len(job.exists)} must exist, {len(job.sha256)} hashes, {len(job.absent)} must not exist")
    print(
        f"gpus: {job.gpu_count or 'no check'}"
        + (f" idle (< {job.gpu_idle_mib} MiB), {'wait' if job.gpu_wait else 'fail if busy'}" if job.gpu_count else "")
    )
    print(
        f"watch: every {job.poll_s:g} s; done /{job.done or 'exit code 0'}/; error /{job.error or '-'}/; progress /{job.progress or '-'}/"
    )
    for spec in job.pulls:
        print(f"pull: {spec.glob} -> {spec.to} (newest {spec.newest}{', while running' if spec.while_running else ''})")
    for entry in job.sync:
        print(f"sync: {len(entry['from'])} paths -> {entry['to']}")
    print(f"--- {job.remote('sh')}\n{launch_script(job)}", end="")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train-relay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="preflight, GPUs, launch, watch, pull")
    p.add_argument("job", type=Path)
    p.add_argument("--attach", action="store_true", help="the job runs already: only watch and pull")
    p.add_argument(
        "--dry-run", action="store_true", help="print what would be done and the launch script; touch nothing"
    )
    p = sub.add_parser("status", help="one look at the job")
    p.add_argument("job", type=Path)
    p = sub.add_parser("pull", help="pull the newest results now (the job may be over or still running)")
    p.add_argument("job", type=Path)
    p = sub.add_parser("stop", help="stop the job on the remote (its whole process group)")
    p.add_argument("job", type=Path)
    a = parser.parse_args(args)
    try:
        job = load(a.job)
        if a.command == "status":
            s = status(job)
            print(f"{job.name} on {job.host}: {verdict(job, s)}" + (f" ({s.progress})" if s.progress else ""))
            return 0
        if a.command == "pull":
            return 0 if pull_ready(job, {}, finished=True) else 1
        if a.command == "stop":
            out = remote(job.host, f"p=$(cat {q(job.remote('pid'))} 2>/dev/null); [ -n \"$p\" ] && kill -0 $p 2>/dev/null "
                                   f"&& kill -TERM -- -$(ps -o pgid= -p $p | tr -d ' ') && echo stopped $p || echo not running")  # fmt: skip
            print(f"{job.name} on {job.host}: {out.strip()}")
            return 0
        if a.dry_run:
            describe(job)
            return 0
        if a.attach:
            if verdict(job, status(job)).startswith("failed: the launcher is gone"):
                raise RelayError(f"{job.name} is not running on {job.host} and left no result; nothing to attach to")
            log(f"attached to {job.name} on {job.host}")
        else:
            preflight(job)
            gpus(job)
            launch(job)
            time.sleep(min(job.poll_s, 60))
        return finish(job, watch(job))
    except RelayError as error:
        log(f"FAILED: {error}")
        return 1
    except KeyboardInterrupt:
        log("interrupted here; the remote job keeps running (train-relay run --attach picks it up again)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
