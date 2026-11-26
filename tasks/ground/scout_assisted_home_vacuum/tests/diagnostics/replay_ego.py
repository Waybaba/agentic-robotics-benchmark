"""Replay only low-level actions in a freshly initialized public apartment."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_sim import Simulation


def replay(path):
    sim = Simulation(617)
    try:
        for line in path.read_text().splitlines():
            sim.advance(json.loads(line))
        return sim.result()
    finally:
        sim.close()


if __name__ == "__main__":
    result = replay(Path(__file__).with_name("onboard") / "success-617.jsonl")
    print(json.dumps(result))
    if not result["success"]:
        raise SystemExit(1)
