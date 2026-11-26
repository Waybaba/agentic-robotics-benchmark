"""Replay a collected solver on 12 freshly sampled homes. No model or host execution of it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess


def main(args):
    episode, task = args.episode.resolve(), args.task.resolve()
    lock = json.loads((episode / "lock.json").read_text())
    assert json.loads((episode / "result.json").read_text())["status"] == "completed"
    commit = subprocess.check_output(["git", "-C", task, "rev-parse", "HEAD"], text=True).strip()
    if lock["task"]["source"].get("commit") != commit:
        raise ValueError("Artifact and task do not have the same verified Git commit")
    area = args.output.resolve()
    area.mkdir(parents=True, exist_ok=False)
    copy = area / task.name
    shutil.copytree(task, copy, ignore=shutil.ignore_patterns("solution", "__pycache__", ".ale-cache"))
    participant = episode / "artifacts" / "0000" / "submission"
    frozen = copy / "solution/frozen"
    frozen.mkdir(parents=True)
    hashes = {}
    for p in sorted(participant.rglob("*")):
        mode = p.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or p.stat().st_size > 4_000_000:
            raise ValueError("Unexpected frozen artifact")
        relative = p.relative_to(participant)
        data = p.read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        dest = frozen / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    assert "run.sh" in hashes
    script = copy / "solution/solve.sh"
    script.write_text('#!/bin/sh\nset -eu\nmkdir -p /home/user/submission\ncp -a "$(dirname "$0")/frozen/." /home/user/submission/\n')
    script.chmod(0o755)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (area / "engine.log").open("w") as log:
        subprocess.run([str(args.ale.resolve()), "run", str(copy), "--agent", "oracle", "--episodes", "2",
                        "--concurrency", "1", "--no-resume", "--episode-retries", "0", "--runs-dir",
                        str(area / "runs"), "--run-id", "frozen", "--set", 'container.provider="docker"'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verdicts = []
    for report in sorted((area / "runs/frozen").glob("*/verification.json")):
        assert json.loads(report.with_name("result.json").read_text())["status"] == "completed"
        verdicts.append(json.loads(report.read_text()))
    assert len(verdicts) == 2 and all(v["metrics"]["episodes"] == 6 for v in verdicts)
    output = {"task_commit": commit, "source_episode": episode.name, "artifact_sha256": hashes,
              "model_calls": 0, "note": "Two diagnostic replays of one frozen program; no second model submission.",
              "verdicts": verdicts}
    (area / "frozen-results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps([v["metrics"] for v in verdicts]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("episode", "task", "ale", "output"):
        ap.add_argument("--" + name, required=True, type=Path)
    main(ap.parse_args())
