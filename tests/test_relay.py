"""Everything runs on this machine (host = "local"): a fake training command writes a log and result files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from train_relay import relay

TRAIN = r"""
for e in 1 2 3; do
  echo "[train] step=$e/3"
  head -c 300000 /dev/urandom > ckpt/slot_$e.pt
  sleep 0.6
done
echo "all done here"
"""


def job_file(tmp: Path, command: str = TRAIN, extra: str = "", pull: str | None = None) -> Path:
    work, home = tmp / "remote", tmp / "home"
    (work / "ckpt").mkdir(parents=True, exist_ok=True)
    (work / "config.yaml").write_text("lr: 1\n")
    home.mkdir(exist_ok=True)
    if pull is None:
        pull = f'[[pull]]\nglob = "{work}/ckpt/slot_*.pt"\nto = "{home}/weights"\nnewest = 2\n'
    path = tmp / "job.toml"
    path.write_text(f"""
name = "t1"
host = "local"
workdir = "{work}"

[launch]
command = '''{command}'''
env = {{ GREETING = "hello there" }}

[watch]
poll_s = 0.2
settle_s = 0
done = 'all done here'
error = 'Traceback|OutOfMemoryError'
progress = 'step=[0-9]+/[0-9]+'

[transfer]
parallel = 4
min_range_mb = 0.05

{pull}
[[sync]]
from = ["{work}/config.yaml"]
to = "{home}"

[hooks]
after_pull = 'echo "$RELAY_FILE" >> {home}/pulled.txt'
on_done = 'echo "$RELAY_STATE $RELAY_NAME" > {home}/end.txt'
on_fail = 'echo "$RELAY_STATE|$RELAY_TEXT" > {home}/end.txt'
{extra}
""")
    return path


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def test_run_watches_and_pulls_the_newest_two(tmp_path, capsys):
    path = job_file(tmp_path)
    assert relay.main(["run", str(path)]) == 0
    work, home = tmp_path / "remote", tmp_path / "home"
    assert sorted(p.name for p in (home / "weights").glob("slot_*")) == ["slot_2.pt", "slot_3.pt"]
    for name in ("slot_2.pt", "slot_3.pt"):
        assert md5(home / "weights" / name) == md5(work / "ckpt" / name)
    record = json.loads((home / "weights/.train-relay.json").read_text())
    assert record["slot_3.pt"]["md5"] == md5(work / "ckpt/slot_3.pt")
    assert (home / "config.yaml").read_text() == "lr: 1\n"
    assert (home / "end.txt").read_text() == "done t1\n"
    assert (home / "pulled.txt").read_text().count("slot_") == 2
    out = capsys.readouterr().out
    assert "running step=" in out and "DONE" in out
    assert "hello there" not in (work / ".train-relay/t1.log").read_text()  # the env is exported, not echoed
    assert not list((home / "weights").glob(".*.part"))

    assert relay.main(["status", str(path)]) == 0
    assert "t1 on local: done (step=3/3)" in capsys.readouterr().out
    before = (home / "weights/slot_3.pt").stat().st_mtime_ns
    assert relay.main(["pull", str(path)]) == 0  # nothing new: nothing is fetched again
    assert (home / "weights/slot_3.pt").stat().st_mtime_ns == before


def test_results_are_pulled_while_the_job_runs(tmp_path):
    work, home = tmp_path / "remote", tmp_path / "home"
    pull = f'[[pull]]\nglob = "{work}/ckpt/slot_*.pt"\nto = "{home}/weights"\nnewest = 5\nwhile_running = true\n'
    slow = TRAIN.replace("sleep 0.6", "sleep 1.2")
    path = job_file(tmp_path, slow, pull=pull)
    job = relay.load(path)
    relay.launch(job)
    seen: dict = {}
    deadline = time.time() + 20
    early = False
    while relay.verdict(job, relay.status(job)) == "running" and time.time() < deadline:
        relay.pull_ready(job, seen, finished=False)
        early = early or (home / "weights/slot_1.pt").exists()
        time.sleep(0.2)
    assert early, "the first result was not pulled while the job was running"
    assert relay.finish(job, relay.verdict(job, relay.status(job))) == 0
    assert len(list((home / "weights").glob("slot_*.pt"))) == 3


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ('echo "Traceback (most recent call last):"; sleep 5', "an error line in the log: Traceback"),
        ("echo working; exit 3", "the command exited with 3"),
        ("echo working", "exited with 0 but the done line never appeared"),
    ],
)
def test_failures_are_named(tmp_path, capsys, command, why):
    path = job_file(tmp_path, command)
    assert relay.main(["run", str(path)]) == 1
    assert why in capsys.readouterr().out
    assert (tmp_path / "home/end.txt").read_text().startswith("failed|t1 on local: failed: ")
    relay.main(["stop", str(path)])


def test_preflight(tmp_path, capsys):
    work = tmp_path / "remote"
    work.mkdir()
    parent = work / "parent.pt"
    parent.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    extra = f'[preflight]\nexists = ["{work}/nope"]\nabsent = ["{parent}"]\n'
    assert relay.main(["run", str(job_file(tmp_path, extra=extra))]) == 1
    out = capsys.readouterr().out
    assert f"missing on the remote: {work}/nope" in out and f"exists already: {parent}" in out
    extra = f'[preflight]\nsha256 = {{ "{parent}" = "{"0" * 64}" }}\n'
    assert relay.main(["run", str(job_file(tmp_path, extra=extra))]) == 1
    assert "sha256 of" in capsys.readouterr().out
    assert not (work / ".train-relay").exists()  # nothing was started
    extra = f'[preflight]\nsha256 = {{ "{parent}" = "{good}" }}\n'
    assert relay.main(["run", str(job_file(tmp_path, extra=extra))]) == 0


def test_gpu_check_and_wait(tmp_path, capsys):
    flag = tmp_path / "busy"
    flag.write_text("1")
    query = f"if [ -e {flag} ]; then echo 30000; echo 12; else echo 5; echo 12; fi"
    extra = f"[gpus]\ncount = 2\nrecheck_s = 0\nquery_command = '{query}'\n"
    assert relay.main(["run", str(job_file(tmp_path, extra=extra))]) == 1
    assert "1 of 2 GPUs idle, the job needs 2" in capsys.readouterr().out
    job = relay.load(job_file(tmp_path, extra=extra.replace("count = 2", "count = 2\nwait = true")))
    subprocess.Popen(["bash", "-c", f"sleep 0.8; rm {flag}"])
    started = time.time()
    relay.gpus(job)
    assert 0.5 < time.time() - started < 10
    assert "waiting for 2 idle GPUs" in capsys.readouterr().out


def test_a_running_job_is_not_started_twice_and_can_be_attached(tmp_path, capsys):
    path = job_file(tmp_path, "sleep 2; echo all done here")
    job = relay.load(path)
    relay.launch(job)
    assert relay.main(["run", str(path)]) == 1
    assert "is running on local already" in capsys.readouterr().out
    assert relay.main(["run", str(path), "--attach"]) == 0
    assert "attached to t1" in capsys.readouterr().out


def test_stop_ends_the_whole_process_group(tmp_path, capsys):
    path = job_file(tmp_path, "sleep 300 & sleep 300; wait")
    job = relay.load(path)
    relay.launch(job)
    time.sleep(0.5)
    pid = int((tmp_path / "remote/.train-relay/t1.pid").read_text())
    children = subprocess.run(["pgrep", "-g", str(os.getpgid(pid))], capture_output=True, text=True).stdout.split()
    assert len(children) >= 3
    assert relay.main(["stop", str(path)]) == 0
    assert "stopped" in capsys.readouterr().out
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-g", children[0]], capture_output=True).returncode != 0
    assert relay.verdict(job, relay.status(job)).startswith("failed: the launcher is gone")


def test_a_corrupt_transfer_is_not_kept(tmp_path, monkeypatch):
    path = job_file(tmp_path, "head -c 200000 /dev/urandom > ckpt/slot_1.pt; echo all done here")
    job = relay.load(path)
    relay.launch(job)
    time.sleep(0.6)
    real = relay.pull_range

    def flipped(job, path, tmp, offset, length):
        ok = real(job, path, tmp, offset, length)
        if offset == 0:
            with open(tmp, "r+b") as f:
                f.write(b"\x00\x01\x02\x03")
        return ok

    monkeypatch.setattr(relay, "pull_range", flipped)
    weights = tmp_path / "home/weights"
    assert relay.pull_ready(job, {}, finished=True) is False
    assert not (weights / "slot_1.pt").exists() and not list(weights.glob(".*.part"))
    monkeypatch.setattr(relay, "pull_range", real)
    assert relay.pull_ready(job, {}, finished=True) is True
    assert md5(weights / "slot_1.pt") == md5(tmp_path / "remote/ckpt/slot_1.pt")


def test_dry_run_and_bad_configs(tmp_path, capsys):
    path = job_file(tmp_path)
    assert relay.main(["run", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "export GREETING='hello there'" in out and "newest 2" in out
    assert not (tmp_path / "remote/.train-relay").exists()
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x y"\nhost = "local"\nworkdir = "/tmp"\n[launch]\ncommand = "true"\n')
    assert relay.main(["status", str(bad)]) == 1
    assert "name may hold" in capsys.readouterr().out
    bad.write_text('name = "x"\nhost = "local"\nworkdir = "/tmp"\ntypo = 1\n[launch]\ncommand = "true"\n')
    assert relay.main(["status", str(bad)]) == 1
    assert "unknown keys ['typo']" in capsys.readouterr().out
