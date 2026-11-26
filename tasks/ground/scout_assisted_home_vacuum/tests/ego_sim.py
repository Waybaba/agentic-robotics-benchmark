"""Indoor scout and vacuum: continuous MuJoCo dynamics and body-mounted RGB only."""

import base64
import io
import math
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import mujoco as mj
import numpy as np
from PIL import Image

from ego_layout import Apartment
from interior import ROOT, add_materials, sweepings

WIDTH, HEIGHT = 160, 112
DT, SUBSTEPS = 0.02, 20
TICKS, FLIGHT = 600, 190
COLORS = {
    "oak": "bea17c",
    "sage": "aea283",
    "hall": "c2b5a4",
    "rose": "aa8c72",
    "slate": "adaaa0",
    "runner": "826374",
}
AMBER, BLUE = ".91 .58 .12 1", ".18 .48 .71 1"


def fmt(v):
    return " ".join(f"{x:.7g}" for x in v)


def block(parent, pos, half, *, mat=None, rgba=None, solid=True, **attr):
    kw = dict(pos=fmt(pos), size=fmt(half), type="box", **attr)
    if mat:
        kw["material"] = mat
    if rgba:
        kw["rgba"] = rgba
    if not solid:
        kw.update(contype="0", conaffinity="0", group="2", mass="0")
    return ET.SubElement(parent, "geom", **kw)


def make_scene(a):
    scene = ET.Element("mujoco", model="apartment")
    ET.SubElement(scene, "compiler", angle="degree")
    ET.SubElement(
        scene,
        "option",
        timestep=str(DT),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        iterations="22",
    )
    vis = ET.SubElement(scene, "visual")
    ET.SubElement(vis, "global", offwidth="800", offheight="480")
    ET.SubElement(vis, "quality", offsamples="1", shadowsize="1024")
    ET.SubElement(
        vis,
        "headlight",
        ambient=".43 .43 .42",
        diffuse=".42 .41 .39",
        specular=".03 .03 .03",
    )
    ET.SubElement(vis, "map", znear=".0015", zfar="4")
    root = ET.SubElement(scene, "worldbody")
    asset = ET.SubElement(scene, "asset")
    files = add_materials(asset, COLORS)
    files["crumbs.obj"] = sweepings()
    ET.SubElement(
        asset,
        "mesh",
        name="crumbs",
        file="crumbs.obj",
        inertia="shell",
        scale="1.7 1.7 1.6",
    )
    ET.SubElement(
        root,
        "light",
        name="day",
        directional="true",
        dir=".36 -.48 -1",
        diffuse=".60 .58 .54",
        specular=".07 .06 .05",
        castshadow="false",
    )
    ET.SubElement(
        root,
        "light",
        directional="true",
        dir="-.28 .33 -1",
        diffuse=".23 .25 .28",
        castshadow="false",
    )
    for room in a.rooms:
        x0, y0, x1, y1 = room.rect
        cx, cy = room.center
        block(
            root,
            (cx, cy, -0.065),
            ((x1 - x0) / 2, (y1 - y0) / 2, 0.065),
            mat="finish_" + room.finish,
        )
        block(
            root,
            (cx, cy, 3.06),
            ((x1 - x0) / 2, (y1 - y0) / 2, 0.07),
            rgba=".92 .91 .87 1",
        )
        block(
            root, (cx, cy, 2.98), (0.52, 0.17, 0.018), rgba=".99 .96 .83 1", solid=False
        )
    for idx, ((x0, y0), (x1, y1), outer) in enumerate(a.segments):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        px, py = (x1 - x0) / 2, (y1 - y0) / 2
        hx, hy = max(0.075, px), max(0.075, py)
        if outer:
            # Recess the pane between a deep sill and lintel; the glazing is solid.
            block(root, (cx, cy, 0.48), (hx, hy, 0.48), mat="plaster")
            block(root, (cx, cy, 2.65), (hx, hy, 0.35), mat="plaster")
            block(
                root,
                (cx, cy, 1.63),
                (hx if px else 0.025, hy if py else 0.025, 0.67),
                rgba=".59 .76 .82 1",
            )
            for f in (-0.66, 0, 0.66):
                block(
                    root,
                    (cx + px * f, cy + py * f, 1.63),
                    (0.045, 0.045, 0.69),
                    rgba=".88 .85 .75 1",
                )
            block(
                root,
                (cx, cy, 0.94),
                (hx + 0.05, hy + 0.05, 0.035),
                rgba=".77 .73 .62 1",
            )
        else:
            block(root, (cx, cy, 1.5), (hx, hy, 1.5), mat="plaster")
        block(
            root,
            (cx, cy, 0.065),
            (hx + 0.016, hy + 0.016, 0.065),
            rgba=".83 .80 .73 1",
            solid=False,
        )
        if not outer and max(hx, hy) > 1.3 and idx % 3 == 0:
            frame = (0.44, 0.115, 0.53) if px else (0.115, 0.44, 0.53)
            face = (0.402, 0.119, 0.489) if px else (0.119, 0.402, 0.489)
            block(root, (cx, cy, 1.62), frame, rgba=".31 .25 .19 1", solid=False)
            block(root, (cx, cy, 1.62), face, mat=f"art{idx % 4}", solid=False)
    for d in a.doors.values():
        x, y = d.xy
        hx, hy = (d.width / 2, 0.075) if d.along_x else (0.075, d.width / 2)
        block(
            root,
            (x, y, (3.0 + d.head) / 2),
            (hx, hy, (3.0 - d.head) / 2),
            mat="plaster",
        )
        block(
            root,
            (x, y, d.head + 0.015),
            (hx + 0.05, hy + 0.04, 0.048),
            rgba=".54 .40 .28 1",
        )
        for sign in (-1, 1):
            off = d.width * 0.5 * sign
            block(
                root,
                (
                    x + (off if d.along_x else 0),
                    y + (0 if d.along_x else off),
                    d.head / 2,
                ),
                (0.055, 0.1, d.head / 2) if d.along_x else (0.1, 0.055, d.head / 2),
                rgba=".59 .46 .34 1",
            )
    mesh_cache = {}

    def furnish(mesh, pt, sz, deg):
        key = (mesh, tuple(sz))
        if key not in mesh_cache:
            filename = mesh + ".obj"
            if filename not in files:
                files[filename] = (ROOT / filename).read_bytes()
            ident = f"f{len(mesh_cache)}"
            ET.SubElement(asset, "mesh", name=ident, file=filename, scale=fmt(sz))
            mesh_cache[key] = ident
        ET.SubElement(
            root,
            "geom",
            type="mesh",
            mesh=mesh_cache[key],
            pos=fmt(pt),
            euler=f"0 0 {deg}",
            material="joinery",
            mass="0",
            contype="0",
            conaffinity="0",
        )

    for f in a.fixtures:
        x, y, z = f.xyz
        w, h, dz = f.size
        furnish(f.mesh, (x, y, 0.008), f.size, f.turn)
        # Invisible collision proxy, with the same footprint and top as the authored mesh.
        hx, hy = f.half_xy
        block(root, (x, y, dz / 2), (hx, hy, dz / 2), rgba="0 0 0 0", group="3")
        if f.mesh == "desk":
            furnish("computerScreen", (x, y, dz + 0.01), (0.53, 0.22, 0.38), f.turn)
        if f.mesh == "tableCoffee":
            furnish("books", (x + 0.055, y, dz + 0.01), (0.33, 0.26, 0.13), f.turn)
    x, y, z = a.shelf.xyz
    top = a.shelf.size[2]
    block(
        root,
        (x, y, top + 0.016),
        (0.22, 0.17, 0.012),
        rgba=".98 .95 .82 1",
        solid=False,
    )
    block(
        root,
        (x, y, top + 0.030),
        (0.15, 0.105, 0.003),
        rgba=BLUE if a.parking == "blue" else AMBER,
        solid=False,
    )
    # Narrow lip on all sides: no line of sight from below the shelf.
    for sg in (-1, 1):
        block(
            root,
            (x + sg * 0.22, y, top + 0.040),
            (0.012, 0.18, 0.043),
            rgba=".47 .34 .22 1",
            solid=False,
        )
        block(
            root,
            (x, y + sg * 0.17, top + 0.040),
            (0.22, 0.012, 0.043),
            rgba=".47 .34 .22 1",
            solid=False,
        )
    for pt, c in (
        (a.dock, ".14 .52 .46 1"),
        (a.stage, ".58 .75 .57 1"),
        (a.park["blue"], BLUE),
        (a.park["amber"], AMBER),
    ):
        block(root, (*pt, 0.009), (0.44, 0.44, 0.009), rgba=c, solid=False)
    block(
        root,
        (a.dock[0], a.dock[1] + 0.30, 0.022),
        (0.09, 0.05, 0.006),
        rgba=".96 .88 .24 1",
        solid=False,
    )
    block(root, (*a.rug, 0.011), (0.88, 0.60, 0.011), mat="textile", solid=False)
    spot_names = {}
    for i, (key, (x, y)) in enumerate(a.spots.items()):
        spot_names[key] = f"dirt{i}"
        ET.SubElement(
            root,
            "geom",
            name=spot_names[key],
            type="mesh",
            mesh="crumbs",
            pos=fmt((x, y, 0.020)),
            rgba=".23 .145 .08 1",
            contype="0",
            conaffinity="0",
            mass="0",
        )

    def planar(name, pt):
        b = ET.SubElement(root, "body", name=name, pos=fmt((*pt, 0)))
        for axis, vec in (("x", "1 0 0"), ("y", "0 1 0"), ("yaw", "0 0 1")):
            ET.SubElement(
                b,
                "joint",
                name=name + "_" + axis,
                type="hinge" if axis == "yaw" else "slide",
                axis=vec,
                damping=(".22" if axis == "yaw" else "3.2")
                if name == "carton"
                else ".02",
                frictionloss=(".08" if axis == "yaw" else ".9")
                if name == "carton"
                else "0",
            )
        return b

    carton = planar("carton", a.box)
    block(
        carton,
        (0, 0, 0.205),
        (0.29, 0.29, 0.205),
        rgba=".65 .47 .24 1",
        mass=".9",
        friction=".6 .01 .01",
    )
    block(
        carton, (0, 0, 0.416), (0.058, 0.29, 0.005), rgba=".82 .65 .40 1", solid=False
    )
    vac = planar("vacuum", a.dock)
    ET.SubElement(
        vac,
        "geom",
        type="cylinder",
        size=".245 .095",
        pos="0 0 .106",
        mass="2.7",
        rgba=".18 .22 .23 1",
    )
    block(
        vac,
        (0, 0.198, 0.120),
        (0.214, 0.064, 0.062),
        rgba=".23 .27 .28 1",
        mass=".10",
        friction=".9 .008 .006",
    )
    block(
        vac, (0, 0.13, 0.238), (0.07, 0.075, 0.048), rgba=".21 .26 .27 1", solid=False
    )
    air = ET.SubElement(root, "body", name="scout", pos=fmt((*a.air_start, 0.29)))
    ET.SubElement(air, "freejoint", name="scout_free")
    ET.SubElement(
        air, "inertial", mass=".36", pos="0 0 0", diaginertia=".0039 .0039 .0065"
    )
    ET.SubElement(
        air, "geom", type="ellipsoid", size=".105 .15 .064", rgba=".22 .27 .29 1"
    )
    for dx in (-0.155, 0.155):
        for dy in (-0.125, 0.125):
            ET.SubElement(
                air,
                "geom",
                type="cylinder",
                size=".082 .014",
                pos=fmt((dx, dy, 0.014)),
                rgba=".29 .34 .35 1",
            )
    for b, name, pt, pitch, fov in (
        (vac, "ground", (0, 0.165, 0.251), 14.0, 72.0),
        (air, "air", (0, 0.162, 0.008), 16.0, 76.0),
    ):
        s, c = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
        ET.SubElement(
            b,
            "camera",
            name=name,
            pos=fmt(pt),
            xyaxes=fmt((1, 0, 0, 0, s, c)),
            fovy=str(fov),
            mode="fixed",
        )
    model = mj.MjModel.from_xml_string(
        ET.tostring(scene, encoding="unicode"), assets=files
    )
    return model, spot_names


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


class Simulation:
    def __init__(self, seed, ticks=TICKS, flight=FLIGHT, order_key=None):
        self.home = Apartment(seed, order_key)
        self.model, names = make_scene(self.home)
        self.data = mj.MjData(self.model)
        self.spot_ids = {k: self.model.geom(v).id for k, v in names.items()}
        self.removed = set()
        self.budget, self.fuel, self.t = ticks, flight, 0
        self.renderer = None
        self.staged = self.scuffed = False
        self.odom = {"vacuum": [0.0] * 3, "drone": [0.0] * 4}
        self.vi, self.ai, self.bi = [
            self.model.body(n).id for n in ("vacuum", "scout", "carton")
        ]
        self.vd = int(self.model.joint("vacuum_x").dofadr[0])
        self.vq = int(self.model.joint("vacuum_x").qposadr[0])
        self.ad = int(self.model.joint("scout_free").dofadr[0])
        self.air_distance = 0.0
        mj.mj_forward(self.model, self.data)

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

    def result(self):
        h = self.home
        protected = h.spots.keys() - h.wanted
        integrity = 1 - len(self.removed & protected) / len(protected)
        clean = len(self.removed & h.wanted) / len(h.wanted)
        parked = (
            self.staged
            and math.dist(self.data.xpos[self.bi, :2], h.park[h.parking]) < 0.43
        )
        docked = math.dist(self.data.xpos[self.vi, :2], h.dock) < 0.46
        success = (
            clean == 1 and integrity == 1 and parked and docked and not self.scuffed
        )
        score = (
            0.45 * clean * integrity
            + 0.125 * (int(self.staged) + int(parked))
            + 0.30 * success
        )
        return {
            "success": bool(success),
            "score": float(score),
            "ticks": self.t,
            "clean_fraction": clean,
            "staged": self.staged,
            "parked": bool(parked),
            "docked": docked,
            "treatment_preserved": integrity,
            "scuffed": self.scuffed,
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
