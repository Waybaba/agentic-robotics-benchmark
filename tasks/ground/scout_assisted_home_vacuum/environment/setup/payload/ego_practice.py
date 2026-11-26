"""Interactive public practice home. Commands and pixels match exam format; no grader."""
import base64
import gzip
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("MUJOCO_GL","osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM","osmesa")
import mujoco as mj
import numpy as np
from PIL import Image
WIDTH,HEIGHT,DT,SUBSTEPS,TICKS,FLIGHT=160,112,0.02,20,600,190

def wrap(rad):
    return (rad + math.pi) % (2 * math.pi) - math.pi

def frame(yaw):
    return np.array((math.sin(yaw), math.cos(yaw))), np.array(
        (math.cos(yaw), -math.sin(yaw))
    )

def encode(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=88, subsampling=0)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def parse(action):
    if not isinstance(action, dict) or set(action) != {"vacuum", "drone"}:
        raise ValueError("expected vacuum and drone")
    g, a = action["vacuum"], action["drone"]
    if (
        not isinstance(g, list)
        or len(g) != 3
        or type(g[2]) is not bool
        or not isinstance(a, list)
        or len(a) != 4
    ):
        raise ValueError("vacuum=[left,right,suction]; drone=[pitch,roll,lift,yaw]")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1
        for v in [*g[:2], *a]
    ):
        raise ValueError("controls must be finite normalized numbers")
    return g, a

class Practice:

    def __init__(self):
        root=Path(__file__).resolve().parent
        meta=json.loads((root/"practice-home.json").read_text())
        with tempfile.NamedTemporaryFile(suffix=".mjb") as fp:
            fp.write(gzip.decompress((root/"practice-home.mjb.gz").read_bytes()));fp.flush()
            self.model=mj.MjModel.from_binary_path(fp.name)
        self.data=mj.MjData(self.model)
        self.home=SimpleNamespace(prompt=meta["instruction"],stage=meta["stage"],
            spots=meta["spots"],rug=meta["rug"])
        self.spot_ids={k:self.model.geom(k).id for k in meta["spots"]}
        self.budget,self.fuel,self.t=TICKS,FLIGHT,0
        self.removed=set();self.staged=self.scuffed=False;self.renderer=None
        self.odom={"vacuum":[0.]*3,"drone":[0.]*4}
        self.vi,self.ai,self.bi=[self.model.body(n).id for n in ("vacuum","scout","carton")]
        self.vd=int(self.model.joint("vacuum_x").dofadr[0])
        self.vq=int(self.model.joint("vacuum_x").qposadr[0])
        self.ad=int(self.model.joint("scout_free").dofadr[0]);self.air_distance=0.
        mj.mj_forward(self.model,self.data)

    def pose(self, air=False):
        idx = self.ai if air else self.vi
        p = self.data.xpos[idx].copy()
        R = self.data.xmat[idx].reshape(3, 3)
        yaw = math.atan2(R[0, 1], R[1, 1])
        return p, yaw

    def view(self, name, width=WIDTH, height=HEIGHT):
        if name not in ("air", "ground"):
            raise ValueError("not an onboard camera")
        if (
            self.renderer is None
            or self.renderer.width != width
            or self.renderer.height != height
        ):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mj.Renderer(self.model, height=height, width=width)
        self.renderer.update_scene(self.data, camera=name)
        return self.renderer.render().copy()

    def observation(self):
        return {
            "instruction": self.home.prompt,
            "ground": encode(self.view("ground")),
            "air": encode(self.view("air")) if self.fuel else None,
            "body_delta": {k: list(v) for k, v in self.odom.items()},
            "left": self.budget - self.t,
            "flight": self.fuel,
        }

    def advance(self, action):
        g, air = parse(action)
        if self.t >= self.budget:
            raise ValueError("episode ended")
        before = [self.pose(False), self.pose(True)]
        for _ in range(SUBSTEPS):
            pos, yaw = self.pose()
            front, right = frame(yaw)
            v = self.data.qvel[self.vd : self.vd + 3]
            drive = (g[0] + g[1]) * 0.77
            omega = (g[0] - g[1]) * 1.43
            force = 2.7 * (
                (drive - float(v[:2] @ front)) * 7 * front
                - float(v[:2] @ right) * 8 * right
            )
            self.data.qfrc_applied[self.vd : self.vd + 2] = np.clip(force, -30, 30)
            self.data.qfrc_applied[self.vd + 2] = np.clip(
                0.12 * 11 * (-omega - v[2]), -1.5, 1.5
            )
            R = self.data.xmat[self.ai].reshape(3, 3)
            up = R[:, 2]
            f, r = frame(self.pose(True)[1])
            pitch, roll, lift, az = air if self.fuel else (0.0, 0.0, -1.0, 0.0)
            target_up = np.array((*(0.35 * (pitch * f + roll * r)), 1.0))
            target_up /= np.linalg.norm(target_up)
            spatial = np.zeros(6)
            mj.mj_objectVelocity(
                self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.ai, spatial, 0
            )
            ang, vel = spatial[:3], spatial[3:]
            thrust = (
                0.36 * 9.81 * (1 + 0.62 * lift) / max(0.73, up[2]) if self.fuel else 0.0
            )
            self.data.xfrc_applied[self.ai, :3] = thrust * up - 0.36 * vel * np.array(
                [1.18, 1.18, 2.1]
            )
            tilt = np.cross(up, target_up)
            self.data.xfrc_applied[self.ai, 3:] = np.clip(
                0.0039 * (54 * tilt - 10 * (ang - up * (ang @ up)))
                + 0.0065 * 9 * (-1.9 * az - ang @ up) * up,
                -0.13,
                0.13,
            )
            mj.mj_step(self.model, self.data)
            vp = self.data.xpos[self.vi, :2]
            bp = self.data.xpos[self.bi, :2]
            if np.linalg.norm(bp - self.home.stage) < 0.42:
                self.staged = True
            if g[2]:
                for key, pt in self.home.spots.items():
                    if key not in self.removed and math.dist(vp, pt) < 0.36:
                        self.removed.add(key)
                        self.model.geom_rgba[self.spot_ids[key], 3] = 0
                if (
                    abs(vp[0] - self.home.rug[0]) < 1.0
                    and abs(vp[1] - self.home.rug[1]) < 0.72
                ):
                    self.scuffed = True
        mj.mj_forward(self.model, self.data)
        self.air_distance += math.dist(self.pose(True)[0], before[1][0])
        for i, (nm, airborne) in enumerate((("vacuum", False), ("drone", True))):
            p, yaw = self.pose(airborne)
            q, old = before[i]
            d = p - q
            f, r = frame(old)
            values = [float(d[:2] @ f), float(d[:2] @ r)]
            if airborne:
                values.append(float(d[2]))
            self.odom[nm] = [round(x, 5) for x in [*values, wrap(yaw - old)]]
        self.t += 1
        self.fuel = max(0, self.fuel - 1)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--wire",action="store_true",help="print the complete evaluation wire format")
    parser.add_argument("--frames",type=Path,help="directory for the current ground.jpg and air.jpg")
    args=parser.parse_args()
    folder=args.frames or Path(tempfile.mkdtemp(prefix="scout-frames-"))
    if not args.wire:folder.mkdir(parents=True,exist_ok=True)
    def emit(sim):
        obs=sim.observation()
        if not args.wire:
            for key in ("ground","air"):
                if obs[key] is not None:
                    path=folder/(key+".jpg")
                    path.write_bytes(base64.b64decode(obs[key]))
                    obs[key]=str(path)
        print(json.dumps(obs),flush=True)
    sim=Practice()
    try:
        emit(sim)
        for line in sys.stdin:
            cmd=json.loads(line)
            if cmd=="close":
                break
            if cmd=="reset":
                sim.close();sim=Practice()
            else:
                sim.advance(cmd)
            emit(sim)
    finally:
        sim.close()
