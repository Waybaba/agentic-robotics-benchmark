"""Unprivileged wall-following bootstrap control, not the map-assisted feasibility oracle."""

import base64
from io import BytesIO
import json
import math
import sys

import numpy as np
from PIL import Image


def image(s):
    return np.asarray(
        Image.open(BytesIO(base64.b64decode(s))).convert("RGB"), dtype=float
    )


class Baseline:
    def __init__(self):
        self.avoid = 0
        self.air_z = 0.29
        self.yaw = 0.0
        self.pos = np.zeros(2)
        self.last_forward = False
        self.visits = {}

    def act(self, t, obs):
        f, r, turn = obs["body_delta"]["vacuum"]
        self.pos += np.array(
            (
                math.sin(self.yaw) * f + math.cos(self.yaw) * r,
                math.cos(self.yaw) * f - math.sin(self.yaw) * r,
            )
        )
        self.yaw += turn
        im = image(obs["ground"])
        self.air_z += obs["body_delta"]["drone"][2]
        if self.last_forward and abs(f) < 0.04:
            self.avoid = 4
        spot = im[66:91, 47:113]
        rr, gg, bb = spot.transpose(2, 0, 1)
        grains = (rr < 126) & (gg < 90) & (rr > 1.22 * gg) & (gg > 1.05 * bb)
        brush = int(grains.sum()) >= 10
        if self.avoid:
            left, right = 0.58, -0.42
            self.avoid -= 1
        else:
            cell = tuple((self.pos / 1.5).round().astype(int))
            self.visits[cell] = self.visits.get(cell, 0) + 1
            steer = 0.22 if self.visits[cell] > 8 else 0.03
            if brush:
                steer = float(
                    np.clip((np.where(grains)[1].mean() - 32) / 90, -0.25, 0.25)
                )
            left, right = 0.65 + steer, 0.65 - steer
        self.last_forward = left + right > 0.6
        lift = float(np.clip((1.35 - self.air_z) * 0.85, -0.48, 0.65))
        air = [0.48, 0.0, lift, 0.32 if t % 17 < 3 else 0.0]
        return {"vacuum": [left, right, brush], "drone": air}


if __name__ == "__main__":
    print("READY", flush=True)
    b = Baseline()
    for line in sys.stdin:
        m = json.loads(line)
        if m["type"] == "act":
            out = {"action": b.act(m["t"], m["obs"])}
        else:
            b = Baseline()
            out = {"ok": True}
        print(json.dumps(out), flush=True)
        if m["type"] == "close":
            break
