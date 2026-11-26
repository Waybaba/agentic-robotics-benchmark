"""Make a reviewer replay. The reference controller only receives the two onboard images."""

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


def make(args):
    task = args.task.resolve()
    sys.path[:0] = [str(task / "tests"), str(task / "tests/diagnostics"), str(task / "solution")]
    from cameras import Cameras
    from controller import Team
    from gallery import font
    from world import Home

    world = Home(args.seed)
    cameras, controller = Cameras(world, overview=True), Team("adaptive")
    frames = []
    bounds = None
    try:
        while True:
            obs = cameras.observation()
            result = world.result()
            done = result["success"] or not obs["left"]
            if world.t % 3 == 0 or done:
                overview = cameras.overview()
                if bounds is None:
                    bounds = overview.convert("L").point(lambda p: 255 if p > 8 else 0).getbbox()
                overhead = overview.crop(bounds)
                overhead.thumbnail((465, 283), Image.Resampling.LANCZOS)
                panel = Image.new("RGB", (798, 384), "#17201f")
                pen = ImageDraw.Draw(panel)
                pen.text((15, 11), "Scout-assisted home vacuum", font=font(24, True), fill="#ebf3ec")
                pen.text((17, 45), "Reviewer's view", font=font(14), fill="#abbbb3")
                panel.paste(overhead, (14 + (465 - overhead.width) // 2, 65 + (278 - overhead.height) // 2))
                for x, key, label in [(484, "air", "Scout"), (641, "ground", "Vacuum")]:
                    pen.text((x, 83), label, font=font(17, True), fill="#e0e9e3")
                    if obs[key]:
                        im = Image.open(BytesIO(base64.b64decode(obs[key]))).convert("RGB")
                        panel.paste(im.resize((140, 140), Image.Resampling.NEAREST), (x, 113))
                    else:
                        pen.rectangle((x, 113, x + 139, 252), fill="#252e2c")
                        pen.text((x + 21, 174), "Flight ended", font=font(15), fill="#aab5ae")
                requested = [world.original[r] for r in (*world.request, "hall")]
                cleaned = sum(len(v - world.dust) for v in requested)
                total = sum(map(len, requested))
                pen.text((484, 276), f"Requested spills: {cleaned}/{total}", font=font(16), fill="#e0e9e3")
                carton = "parked in hall" if result["parked"] else "first delivery made" if result["staged"] else "first delivery pending"
                pen.text((484, 301), "Carton: " + carton, font=font(16), fill="#e0e9e3")
                label = "All work complete" if result["success"] else f"Scout battery: {obs['flight']} ticks"
                pen.text((16, 350), f"Tick {world.t:03d}   |   {label}", font=font(17, True), fill="#99cfaf")
                frames.append(panel)
            if done:
                break
            world.advance(controller.choose(world.t, obs))
    finally:
        cameras.close()
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=[190] * (len(frames) - 1) + [1700], loop=0, optimize=False)
    print(f"{len(frames)} frames; ticks={world.t}; full_success={result['success']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--seed", type=int, default=617)
    p.add_argument("--output", type=Path, required=True)
    make(p.parse_args())
