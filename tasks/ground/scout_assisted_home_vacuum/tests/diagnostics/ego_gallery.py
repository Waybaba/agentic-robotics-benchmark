"""Fly the public scout, saving real onboard JPEGs; no overview or camera teleports."""

import argparse
import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ego_reference import Navigator, Simulation, drone_controls, wrap


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    sim = Simulation(617)
    h = sim.home
    cases = [
        ("Living room", "loungeSofaLong", h.living),
        ("Northern bedroom", "bedDouble", h.bedrooms[1]),
        ("Study work slip", "bookcaseOpen", h.study),
    ]
    nav = Navigator(h, True)
    panels = []
    try:
        for name, mesh, room in cases:
            f = next(f for f in h.fixtures if f.room == room and f.mesh == mesh)
            focus = np.asarray(f.xyz[:2])
            center = np.asarray(h.rooms[room].center)
            towards = center - focus
            towards /= np.linalg.norm(towards)
            eye = focus + towards * (1.24 if name == "Study work slip" else 2.70)
            route = nav.path(sim.pose(True)[0][:2], eye)
            while sim.fuel:
                pos, yaw = sim.pose(True)
                while route and math.dist(pos[:2], route[0]) < 0.34:
                    route.pop(0)
                aim = route[0] if route else eye
                d = (aim if route else focus) - pos[:2]
                heading = math.atan2(d[0], d[1])
                z = 1.32 if route else (2.12 if name == "Study work slip" else 1.53)
                if (
                    not route
                    and math.dist(pos[:2], eye) < 0.19
                    and abs(z - pos[2]) < 0.10
                    and abs(wrap(heading - yaw)) < 0.08
                ):
                    obs = sim.observation()
                    raw = base64.b64decode(obs["air"])
                    slug = name.lower().replace(" ", "-")
                    (out / (slug + ".jpg")).write_bytes(raw)
                    panels.append(
                        (name, sim.t, Image.open(io.BytesIO(raw)).convert("RGB"))
                    )
                    break
                if sim.t < 6 and pos[2] < 0.98:
                    aim = pos[:2]
                ctrl = drone_controls(sim, aim, z, heading)
                sim.advance({"vacuum": [0, 0, False], "drone": ctrl})
            else:
                raise RuntimeError("survey exceeded the same scout battery as the task")
        canvas = Image.new("RGB", (984, 277), "#202628")
        draw = ImageDraw.Draw(canvas)
        for i, (name, tick, im) in enumerate(panels):
            x = 8 + i * 328
            canvas.paste(im.resize((320, 224), Image.Resampling.LANCZOS), (x, 8))
            draw.text((x + 5, 239), f"{name}  |  tick {tick}", fill="#f0eee5")
        canvas.save(out / "egocentric-scout.jpg", quality=92)
        print([(name, t) for name, t, _ in panels], flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
