"""Replay only low-level actions in a freshly initialized public apartment."""

import argparse
import base64
import io
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_sim import Simulation


def replay(path, gif=None):
    sim = Simulation(617)
    frames = []
    try:
        for line in path.read_text().splitlines():
            if gif and sim.t % 3 == 0:
                obs = sim.observation()
                panel = Image.new("RGB", (660, 261), "#21282a")
                draw = ImageDraw.Draw(panel)
                for x, key, label in ((7, "ground", "Vacuum"), (333, "air", "Scout")):
                    draw.text((x + 4, 8), f"{label}  |  tick {sim.t}", fill="#eeeae1")
                    if obs[key]:
                        with Image.open(io.BytesIO(base64.b64decode(obs[key]))) as im:
                            panel.paste(
                                im.resize((320, 224), Image.Resampling.LANCZOS), (x, 30)
                            )
                    else:
                        draw.text((x + 90, 124), "Scout camera off", fill="#c7c4bb")
                frames.append(panel)
            sim.advance(json.loads(line))
        verdict = sim.result()
        if gif and verdict["success"]:
            frames[0].save(
                gif,
                save_all=True,
                append_images=frames[1:],
                duration=150,
                loop=0,
                optimize=True,
                disposal=2,
            )
        return verdict
    finally:
        sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gif", type=Path, help="Optional accelerated replay of actual wire JPEGs"
    )
    result = replay(
        Path(__file__).with_name("onboard") / "success-617.jsonl", p.parse_args().gif
    )
    print(json.dumps(result))
    if not result["success"]:
        raise SystemExit(1)
