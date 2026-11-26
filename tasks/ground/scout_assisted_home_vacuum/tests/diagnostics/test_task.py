import base64
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import mujoco as mj
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "tests"))
from ego_layout import Apartment
from ego_sim import Simulation, parse

STOP = {"vacuum": [0, 0, False], "drone": [0, 0, 0, 0]}


class Layouts(unittest.TestCase):
    def test_graph_and_language(self):
        topologies = set()
        sizes = set()
        cycle_counts = set()
        for seed in range(80):
            h = Apartment(seed)
            sizes.add(len(h.rooms))
            self.assertGreater(h.width * h.depth, 325)
            self.assertGreaterEqual(len(h.doors), len(h.rooms))
            cycle_counts.add(len(h.doors) - len(h.rooms) + 1)
            seen = {0}
            for _ in h.rooms:
                for a, b in h.doors:
                    if a in seen or b in seen:
                        seen.update((a, b))
            self.assertEqual(len(seen), len(h.rooms))
            topologies.add(tuple(sorted(h.doors)))
            beds = sorted(
                (r for r in h.rooms if r.kind == "bedroom"), key=lambda r: r.center[1]
            )
            self.assertEqual(len(beds), 2)
            self.assertEqual(
                h.requested_bedroom, beds[1 if "northern bedroom" in h.prompt else 0].id
            )
            self.assertEqual(len(h.wanted), 5)
            for key, pt in h.spots.items():
                if key == "under-carton":
                    continue
                room, side, j = key
                x0, y0, x1, y1 = h.rooms[room].rect
                dist = {
                    "north": y1 - pt[1],
                    "south": pt[1] - y0,
                    "east": x1 - pt[0],
                    "west": pt[0] - x0,
                }
                self.assertEqual(min(dist, key=dist.get), side)
                self.assertEqual(h.room_at(pt), room)
            self.assertTrue(
                any(
                    f.mesh == "bedDouble" and f.room == h.requested_bedroom
                    for f in h.fixtures
                )
            )
        self.assertEqual(sizes, {9, 12})
        self.assertEqual(cycle_counts, {1, 2, 3, 4})
        self.assertGreater(len(topologies), 80 // 2)

    def test_same_home_different_meaning(self):
        xs = [Apartment(617, order_key=k) for k in range(32)]
        self.assertTrue(
            all(
                h.spots == xs[0].spots
                and h.rooms == xs[0].rooms
                and h.doors == xs[0].doors
                for h in xs
            )
        )
        self.assertGreater(len({frozenset(h.wanted) for h in xs}), 3)
        self.assertEqual({h.parking for h in xs}, {"blue", "amber"})

    def test_controls_are_numbers_not_waypoints(self):
        for bad in (
            ["E", "N", 2, True],
            {**STOP, "map": 1},
            {**STOP, "drone": [0, 0, float("nan"), 0]},
            {**STOP, "vacuum": [1.1, 0, False]},
            {**STOP, "drone": [0, 0, True, 0]},
        ):
            with self.assertRaises(ValueError):
                parse(bad)


class CamerasAndPhysics(unittest.TestCase):
    def setUp(self):
        self.s = Simulation(617)

    def tearDown(self):
        self.s.close()

    def test_perspective_and_interface(self):
        s = self.s
        obs = s.observation()
        self.assertEqual(
            set(obs), {"instruction", "ground", "air", "body_delta", "left", "flight"}
        )
        for name, body in (("ground", "vacuum"), ("air", "scout")):
            cid = s.model.camera(name).id
            self.assertFalse(s.model.cam_orthographic[cid])
            self.assertEqual(s.model.cam_bodyid[cid], s.model.body(body).id)
            fwd = -s.data.cam_xmat[cid].reshape(3, 3)[:, 2]
            self.assertGreater(fwd[1], 0.85)
            self.assertGreater(fwd[2], -0.45)
            im = Image.open(io.BytesIO(base64.b64decode(obs[name])))
            self.assertEqual(im.size, (160, 112))
            self.assertGreater(float(np.std(im)), 12)
        self.assertEqual(s.result()["score"], 0.0)

    def test_real_vertical_yaw_and_ceiling(self):
        s = self.s
        first = s.view("air")
        pos0 = s.pose(True)[0].copy()
        for _ in range(3):
            s.advance({"vacuum": [0, 0, False], "drone": [0, 0, 0.55, 0.70]})
        pos, yaw = s.pose(True)
        self.assertGreater(pos[2] - pos0[2], 0.55)
        self.assertGreater(abs(yaw), 0.5)
        self.assertGreater(np.abs(first.astype(float) - s.view("air")).mean(), 8)
        for _ in range(15):
            s.advance({"vacuum": [0, 0, False], "drone": [0, 0, 1, 0]})
        self.assertLess(s.pose(True)[0][2], 3.01)
        self.assertGreater(s.pose(True)[0][2], 2.65)
        self.assertLess(np.linalg.norm(s.pose()[0][:2] - s.home.dock), 0.02)

    def test_body_forward_after_turn(self):
        s = self.s
        for _ in range(2):
            s.advance({"vacuum": [1, -1, False], "drone": [0, 0, 0, 0]})
        pos, yaw = s.pose()
        old = pos.copy()
        for _ in range(2):
            s.advance({"vacuum": [0.45, 0.45, False], "drone": [0, 0, 0, 0]})
        delta = s.pose()[0][:2] - old[:2]
        self.assertGreater(
            float(delta @ np.array((math.sin(yaw), math.cos(yaw)))), 0.30
        )
        self.assertGreater(abs(delta[0]), 0.22)

    def test_shelf_color_not_observable_below(self):
        a = self.s
        k = next(k for k in range(30) if Apartment(617, k).parking != a.home.parking)
        b = Simulation(617, order_key=k)
        try:
            np.testing.assert_array_equal(a.view("ground"), b.view("ground"))
            for sim in (a, b):
                s = sim.home.shelf
                x, y = s.xyz[:2]
                c = np.asarray(sim.home.rooms[sim.home.study].center)
                v = c - np.array((x, y))
                v /= np.linalg.norm(v)
                eye = np.array((x, y)) + v * 1.25
                yaw = math.atan2(-v[0], -v[1])
                q = int(sim.model.joint("scout_free").qposadr[0])
                sim.data.qpos[q : q + 7] = (
                    *eye,
                    2.12,
                    math.cos(yaw / 2),
                    0,
                    0,
                    -math.sin(yaw / 2),
                )
                first = sim.model.body("vacuum").pos
                sim.data.qpos[sim.vq : sim.vq + 3] = (*(eye - first[:2]), -yaw)
                mj.mj_forward(sim.model, sim.data)
            self.assertGreater(np.count_nonzero(a.view("air") != b.view("air")), 20)
            np.testing.assert_array_equal(a.view("ground"), b.view("ground"))
        finally:
            b.close()

    def test_public_practice_parity(self):
        path = TASK / "environment/setup/payload/ego_practice.py"
        spec = importlib.util.spec_from_file_location("public_ego_rig", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        p = mod.Practice()
        try:
            for a in (
                STOP,
                {"vacuum": [0.4, 0.2, False], "drone": [0.2, -0.1, 0.5, 0.25]},
            ):
                self.s.advance(a)
                p.advance(a)
                self.assertEqual(self.s.observation(), p.observation())
        finally:
            p.close()


if __name__ == "__main__":
    unittest.main()
