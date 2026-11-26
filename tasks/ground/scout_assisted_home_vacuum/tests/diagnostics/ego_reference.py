"""Author-only map-assisted feasibility pilot. Never imported by the verifier or solver."""

import argparse
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_sim import Simulation, frame, wrap


class Navigator:
    def __init__(self, home, air=False):
        self.res = 0.30
        self.home = home
        self.x, self.y = np.meshgrid(
            np.arange(0, home.width + 0.3, 0.3), np.arange(0, home.depth + 0.3, 0.3)
        )
        x, y = self.x, self.y
        margin = 0.34 if air else 0.36
        self.clear = (
            (x > margin)
            & (x < home.width - margin)
            & (y > margin)
            & (y < home.depth - margin)
        )
        for (x0, y0), (x1, y1), _ in home.segments:
            self.clear &= (
                (x < x0 - margin - 0.08)
                | (x > x1 + margin + 0.08)
                | (y < y0 - margin - 0.08)
                | (y > y1 + margin + 0.08)
            )
        for f in home.fixtures:
            if air and f.size[2] < 0.97:
                continue
            fx, fy, _ = f.xyz
            hx, hy = f.half_xy
            self.clear &= (abs(x - fx) > hx + margin) | (abs(y - fy) > hy + margin)

    def path(self, start, goal, box=None):
        clear = self.clear
        if box is not None:
            clear = clear & (
                (abs(self.x - box[0]) > 0.57) | (abs(self.y - box[1]) > 0.57)
            )

        def snap(pt):
            distance = (self.x - pt[0]) ** 2 + (self.y - pt[1]) ** 2
            j, i = np.unravel_index(
                np.argmin(np.where(clear, distance, np.inf)), clear.shape
            )
            return int(i), int(j)

        source, end = snap(start), snap(goal)
        todo = [(0, 0, source)]
        prev = {source: None}
        cost = {source: 0}
        while todo:
            _, d, (i, j) = heapq.heappop(todo)
            if (i, j) == end:
                break
            if d > cost[i, j] + 1e-9:
                continue
            for dx, dy in (
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                k, l = i + dx, j + dy
                if not (
                    0 <= l < clear.shape[0] and 0 <= k < clear.shape[1] and clear[l, k]
                ):
                    continue
                if dx and dy and not (clear[j, k] and clear[l, i]):
                    continue
                q = k, l
                nd = d + (1.414214 if dx and dy else 1)
                if nd < cost.get(q, float("inf")) - 1e-8:
                    prev[q] = (i, j)
                    cost[q] = nd
                    heapq.heappush(
                        todo, (nd + math.hypot(k - end[0], l - end[1]), nd, q)
                    )
        if end not in prev:
            raise RuntimeError(f"no path from {start} to {goal}")
        path = []
        q = end
        while q is not None:
            path.append(q)
            q = prev[q]
        path.reverse()

        def line_ok(a, b):
            n = 2 * max(abs(a[0] - b[0]), abs(a[1] - b[1]), 1)
            return all(
                clear[
                    round(a[1] + (b[1] - a[1]) * k / n),
                    round(a[0] + (b[0] - a[0]) * k / n),
                ]
                for k in range(n + 1)
            )

        slim = []
        i = 0
        while i < len(path) - 1:
            j = i + 1
            while j + 1 < len(path) and line_ok(path[i], path[j + 1]):
                j += 1
            slim.append(np.array(path[j]) * self.res)
            i = j
        slim.append(np.array(goal))
        return slim


def wheels(p, yaw, target, slow=1.0):
    v = np.asarray(target) - p[:2]
    err = wrap(math.atan2(v[0], v[1]) - yaw)
    f = min(1.28 * slow, 1.55 * np.linalg.norm(v)) * max(0.0, math.cos(err)) ** 4
    turn = float(np.clip(2.05 * err, -2.4, 2.4))
    l, r = f / 1.54 + turn / 2.86, f / 1.54 - turn / 2.86
    m = max(1.0, abs(l), abs(r))
    return [float(l / m), float(r / m)]


def drone_controls(sim, goal, z, yaw_target):
    p, yaw = sim.pose(True)
    f, r = frame(yaw)
    v = sim.data.qvel[sim.ad : sim.ad + 3]
    wanted = np.clip((np.asarray(goal) - p[:2]) * 1.45, -1.75, 1.75)
    accel = np.clip((wanted - v[:2]) * 1.45 + 1.18 * wanted, -3.0, 3.0)
    vz = float(np.clip((z - p[2]) * 1.55, -0.9, 0.9))
    return [
        float(np.clip(accel @ f / 3.40, -1, 1)),
        float(np.clip(accel @ r / 3.40, -1, 1)),
        float(np.clip((2.4 * (vz - v[2]) + 2.1 * vz) / 6.08, -1, 1)),
        float(np.clip(wrap(yaw_target - yaw) * 1.6 / 1.9, -1, 1)),
    ]


def read_slip(sim):
    """Read actual shelf pixels; the privileged pilot knows where to point, not the ink."""
    s = sim.home.shelf
    pt = np.array([*s.xyz[:2], s.size[2] + 0.033])
    cid = sim.model.camera("air").id
    rot = sim.data.cam_xmat[cid].reshape(3, 3)
    xyz = rot.T @ (pt - sim.data.cam_xpos[cid])
    if xyz[2] >= -0.04:
        return None
    im = sim.view("air")
    h, w = im.shape[:2]
    f = (h / 2) / math.tan(math.radians(sim.model.cam_fovy[cid]) / 2)
    u = int(round(w / 2 + xyz[0] * f / -xyz[2]))
    v = int(round(h / 2 - xyz[1] * f / -xyz[2]))
    if not (3 < u < w - 3 and 3 < v < h - 3):
        return None
    rgb = np.median(im[v - 2 : v + 3, u - 2 : u + 3].reshape(-1, 3), axis=0)
    if rgb[0] > rgb[2] * 2.0 and rgb[1] > rgb[2] * 1.45:
        return "amber"
    if rgb[2] > rgb[0] * 1.55 and rgb[1] > rgb[0] * 1.22:
        return "blue"
    return None


class Pilot:
    def __init__(self, sim):
        self.sim = sim
        h = sim.home
        self.nav = Navigator(h)
        self.airnav = Navigator(h, True)
        self.route = []
        self.target = None
        self.job = "approach"
        self.read = None
        self.active = None
        self.align = None
        self.u = np.asarray(h.axis, dtype=float)
        self.shelf = np.array(h.shelf.xyz[:2])
        center = np.array(h.rooms[h.study].center)
        self.eye = (
            self.shelf
            + (center - self.shelf) / np.linalg.norm(center - self.shelf) * 1.28
        )
        self.airpath = self.airnav.path(sim.pose(True)[0][:2], self.eye)
        self.log = {}

    def move(self, goal, box=True, tol=0.17):
        s = self.sim
        p, yaw = s.pose()
        goal = np.asarray(goal)
        if math.dist(p[:2], goal) < tol:
            self.target = None
            self.route = []
            return None
        if self.target != tuple(goal):
            self.route = self.nav.path(
                p[:2], goal, s.data.xpos[s.bi, :2] if box else None
            )
            self.target = tuple(goal)
        while len(self.route) > 1 and math.dist(p[:2], self.route[0]) < 0.24:
            self.route.pop(0)
        return wheels(p, yaw, self.route[0] if self.route else goal)

    def aerial(self):
        s = self.sim
        p, _ = s.pose(True)
        if not s.fuel or self.read:
            return [0.0, 0.0, 0.0, 0.0]
        if p[2] < 1.12 and s.t < 8:
            return drone_controls(s, p[:2], 1.32, 0.0)
        while self.airpath and math.dist(p[:2], self.airpath[0]) < 0.40:
            self.airpath.pop(0)
        target = self.airpath[0] if self.airpath else self.eye
        look = (target if self.airpath else self.shelf) - p[:2]
        yaw = math.atan2(look[0], look[1])
        if not self.airpath and math.dist(p[:2], self.eye) < 0.43 and p[2] > 1.90:
            self.read = read_slip(s)
            if self.read:
                self.log["slip_tick"] = s.t
        return drone_controls(s, target, 1.31 if self.airpath else 2.15, yaw)

    def ground(self):
        s = self.sim
        h = s.home
        p, yaw = s.pose()
        b = s.data.xpos[s.bi, :2].copy()
        cmd = [0.0, 0.0]
        brush = False
        if self.job in ("approach", "reverse"):
            sign = 1 if self.job == "approach" else -1
            if self.align is None:
                self.align = b - self.u * sign * 0.94
            pos = self.align
            m = self.move(pos, tol=0.040)
            if m is None:
                d = sign * self.u
                err = wrap(math.atan2(d[0], d[1]) - yaw)
                if abs(err) > 0.024:
                    turn = float(np.clip(err * 0.8, -0.64, 0.64))
                    cmd = [turn, -turn]
                else:
                    self.job = "push" if sign == 1 else "return-push"
                    self.align = None
            else:
                cmd = m
        elif self.job in ("push", "return-push"):
            inbound = self.job == "return-push"
            # Vacuum the revealed foyer spill during the outward push, before the
            # final parking position constrains access to this narrow doorway.
            brush = not inbound
            if inbound and self.read is None:
                return [0.0, 0.0, False]
            target = np.array(h.park[self.read] if inbound else h.stage)
            direction = (-1 if inbound else 1) * self.u
            remaining = float((target - b) @ direction)
            if (inbound and s.result()["parked"] and remaining < 0.24) or (
                not inbound and s.staged and remaining < 0.10
            ):
                self.job = "clean" if inbound else "reverse"
                self.target = None
                self.log["park" if inbound else "stage"] = s.t
            else:
                normal = np.array([-direction[1], direction[0]])
                lateral = float((target - p[:2]) @ normal)
                aim = direction + np.clip(lateral * 1.10, -0.25, 0.25) * normal
                err = wrap(math.atan2(aim[0], aim[1]) - yaw)
                v = 0.32 if remaining < 0.65 else 0.67
                turn = float(np.clip(err * 0.68, -0.28, 0.28))
                cmd = [v / 1.54 + turn, v / 1.54 - turn]
        elif self.job == "clean":
            remaining = h.wanted - s.removed
            if not remaining:
                self.job = "dock"
                self.target = None
            else:
                if self.active not in remaining:
                    self.active = min(
                        remaining, key=lambda k: (math.dist(p[:2], h.spots[k]), str(k))
                    )
                    self.target = None
                pt = h.spots[self.active]
                m = self.move(pt, tol=0.14)
                cmd = m or [0.0, 0.0]
                brush = math.dist(p[:2], pt) < 0.34
        else:
            cmd = self.move(h.dock, tol=0.19) or [0.0, 0.0]
        return [*cmd, brush]

    def act(self):
        return {"drone": self.aerial(), "vacuum": self.ground()}


def run(seed, directory=None, trace=False, actions_out=None):
    sim = Simulation(seed)
    pilot = Pilot(sim)
    capture = directory is not None
    if capture:
        directory.mkdir(exist_ok=True, parents=True)
    h = sim.home
    actions = []
    seen = set()
    history = []
    try:
        for tick in range(sim.budget):
            act = pilot.act()
            if actions_out is not None:
                actions.append(act)
            if capture:
                ap = sim.pose(True)[0]
                rid = h.room_at(ap[:2])
                label = None
                if tick == 0:
                    label = "dock"
                elif tick == pilot.log.get("slip_tick"):
                    label = "slip"
                elif rid is not None and ap[2] > 1.0 and (rid not in seen or tick == 8):
                    label = "room-" + str(rid)
                    seen.add(rid)
                elif tick % 32 == 0 and tick < sim.fuel:
                    label = f"flight-{tick}"
                if label:
                    for cam in ("air", "ground"):
                        Image.fromarray(sim.view(cam)).save(
                            directory / f"{label}-{cam}.png"
                        )
                        Image.fromarray(sim.view(cam, 640, 448)).save(
                            directory / f"{label}-{cam}-review.png"
                        )
            sim.advance(act)
            if trace and tick % 10 == 0:
                history.append(
                    dict(
                        t=tick,
                        job=pilot.job,
                        ground=np.round(sim.pose()[0][:2], 2).tolist(),
                        box=np.round(sim.data.xpos[sim.bi, :2], 2).tolist(),
                        air=np.round(sim.pose(True)[0], 2).tolist(),
                        remaining=len(h.wanted - sim.removed),
                        current=str(pilot.active),
                        read=pilot.read,
                        waypoint=pilot.route[0].tolist() if pilot.route else None,
                    )
                )
            if sim.result()["success"]:
                break
        if actions_out is not None:
            actions_out.mkdir(parents=True, exist_ok=True)
            (actions_out / f"success-{seed}.jsonl").write_text(
                "".join(json.dumps(a, separators=(",", ":")) + "\n" for a in actions)
            )
        return dict(
            seed=seed,
            rooms=len(h.rooms),
            area=round(h.width * h.depth, 1),
            doors=len(h.doors),
            read=pilot.read,
            expected=h.parking,
            mission=h.prompt,
            **sim.result(),
            events=pilot.log,
            **({"trace": history} if trace else {}),
        )
    finally:
        sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument(
        "--actions", type=Path, help="export the actual action sequence for replay"
    )
    args = ap.parse_args()
    for seed in args.seeds:
        print(
            json.dumps(
                run(
                    seed,
                    args.capture / str(seed) if args.capture else None,
                    args.trace,
                    args.actions,
                )
            ),
            flush=True,
        )
