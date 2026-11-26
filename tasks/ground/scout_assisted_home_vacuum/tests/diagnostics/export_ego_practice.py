"""Publish exactly one practice apartment; never package a hidden generator or scorer."""

import argparse
import gzip
import inspect
import json
from pathlib import Path
import sys
import tempfile
import textwrap

import mujoco as mj
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_sim import (
    Simulation,
    wrap,
    frame,
    encode,
    parse,
    WIDTH,
    HEIGHT,
    DT,
    SUBSTEPS,
    TICKS,
    FLIGHT,
)

HEADER = '''"""Interactive public practice home. Commands and pixels match exam format; no grader."""
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
'''
INIT = """
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
"""
MAIN = """
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
"""


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    sim = Simulation(617)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scene.mjb"
            mj.mj_saveModel(sim.model, str(p))
            (out / "practice-home.mjb.gz").write_bytes(
                gzip.compress(p.read_bytes(), compresslevel=7, mtime=0)
            )
        names = {
            sim.model.geom(gid).name: sim.home.spots[k]
            for k, gid in sim.spot_ids.items()
        }
        meta = {
            "public_seed": 617,
            "instruction": sim.home.prompt,
            "stage": sim.home.stage,
            "rug": sim.home.rug,
            "spots": names,
        }
        (out / "practice-home.json").write_text(
            json.dumps(meta, separators=(",", ":")) + "\n"
        )
        constants = f"WIDTH,HEIGHT,DT,SUBSTEPS,TICKS,FLIGHT={WIDTH},{HEIGHT},{DT},{SUBSTEPS},{TICKS},{FLIGHT}\n"
        code = (
            HEADER
            + constants
            + "\n"
            + "\n".join(inspect.getsource(f) for f in (wrap, frame, encode, parse))
        )
        code += "\nclass Practice:\n" + INIT
        for name in ("pose", "view", "observation", "advance", "close"):
            code += "\n" + textwrap.indent(
                textwrap.dedent(inspect.getsource(getattr(Simulation, name))), "    "
            )
        (out / "ego_practice.py").write_text(code + MAIN)
        Image.fromarray(sim.view("ground")).save(out / "ground-start.png")
        Image.fromarray(sim.view("air")).save(out / "scout-start.png")
    finally:
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    export(parser.parse_args().output)
