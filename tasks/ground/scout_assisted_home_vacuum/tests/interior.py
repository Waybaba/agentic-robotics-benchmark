"""Render the same furniture meshes and floor finishes in viewer and robot cameras."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).with_name("furnishings")
SURFACES = Path(__file__).with_name("interior_materials")
MATERIALS = """936944 684936 829599 343e45 69767a bfcaca
3c6864 304f59 e8dfce 879d9e 507d55 eecb82 bcb6a2 8d8c80""".split()
HEIGHT = {
    "bedDouble": 0.65,
    "cabinetBedDrawer": 0.64,
    "lampRoundFloor": 1.35,
    "loungeSofaLong": 0.8,
    "tableCoffee": 0.42,
    "loungeChair": 0.76,
    "bookcaseOpen": 1.53,
    "desk": 0.80,
    "chairDesk": 0.84,
    "plantSmall1": 0.74,
    "kitchenSink": 0.80,
    "kitchenStove": 0.82,
    "kitchenFridge": 1.58,
    "kitchenCabinetDrawer": 0.82,
    "computerScreen": 0.38,
    "books": 0.16,
}


def png(arr):
    out = BytesIO()
    Image.fromarray(arr.astype("uint8")).save(out, "PNG")
    return out.getvalue()


@lru_cache(maxsize=12)
def finish(room, swatch):
    base = np.asarray(list(bytes.fromhex(swatch)))
    name = {
        "oak": "herringbone_parquet",
        "rose": "laminate_floor_03",
        "hall": "terrazzo_tiles",
        "slate": "terrazzo_tiles",
        "sage": "herringbone_parquet",
        "plaster": "painted_plaster_wall",
        "veneer": "oak_veneer_02",
        "runner": "denim_fabric",
    }[room]
    with Image.open(SURFACES / (name + ".jpg")) as im:
        raw = np.asarray(im, dtype=float)
    norm = raw / np.maximum(raw.mean(axis=(0, 1)), 1)
    return png(np.clip(base * (0.32 + 0.68 * norm.clip(0.60, 1.35)), 0, 255))


@lru_cache(maxsize=1)
def furniture_atlas():
    out = Image.new("RGB", (512, 512))
    for i, swatch in enumerate(MATERIALS):
        typ = "veneer" if i in (0, 1) else "runner" if i in (6, 7, 8, 12) else None
        tile = (
            Image.open(BytesIO(finish(typ, swatch))).resize(
                (128, 128), Image.Resampling.LANCZOS
            )
            if typ
            else Image.new("RGB", (128, 128), "#" + swatch)
        )
        out.paste(tile, ((i % 4) * 128, (i // 4) * 128))
    return png(np.asarray(out))


@lru_cache(maxsize=1)
def sweepings():
    # A thin irregular patch and scattered grains, not a pile of rigid cylinders.
    rng = np.random.default_rng(4311)
    verts = []
    faces = []

    def flake(x, y, a, b, n):
        i = len(verts) + 1
        verts.append((x, y, 0.0025))
        for angle in np.arange(n) * 2 * np.pi / n:
            jitter = rng.uniform(0.78, 1.2)
            verts.append(
                (x + np.cos(angle) * a * jitter, y + np.sin(angle) * b * jitter, 0.0003)
            )
        for j in range(n):
            faces.append((i, i + 1 + j, i + 1 + (j + 1) % n))

    flake(0, 0, 0.075, 0.052, 13)
    for _ in range(180):
        x, y = np.clip(rng.normal(size=2), -2.5, 2.5) * (0.111, 0.079)
        s = rng.uniform(0.003, 0.020)
        flake(x, y, s, 0.64 * s, 5)
    return (
        "\n".join(
            ["v %.5f %.5f %.5f" % v for v in verts] + ["f %s %s %s" % f for f in faces]
        )
        + "\n"
    ).encode()


@lru_cache(maxsize=4)
def print_texture(index):
    im = Image.new(
        "RGB", (320, 400), ("#efe5cd", "#e0e7dd", "#ede3da", "#e6d8c5")[index]
    )
    pencil = ImageDraw.Draw(im)
    inks = (
        ("#bf6848", "#366761", "#c99b5e"),
        ("#688b87", "#405669", "#bb9460"),
        ("#a67173", "#354e5a", "#d2a66f"),
        ("#af7149", "#525a48", "#cbb28e"),
    )[index]
    pencil.ellipse((140, 44, 238, 142), fill=inks[2])
    pencil.polygon(
        [
            (20, 292),
            (20, 244),
            (111, 144),
            (186, 217),
            (245, 165),
            (302, 254),
            (302, 292),
        ],
        fill=inks[0],
    )
    pencil.polygon(
        [(20, 313), (20, 266), (130, 220), (225, 258), (302, 201), (302, 313)],
        fill=inks[1],
    )
    for y in (332, 340):
        pencil.line((33, y, 135 if y == 332 else 99, y), fill=inks[1], width=2)
    return png(np.asarray(im))


def add_materials(asset, colors):
    files = {}

    def mat(name, data, repeat="1 1", uniform="false"):
        filename = name + ".png"
        files[filename] = data
        ET.SubElement(asset, "texture", name=name, type="2d", file=filename)
        ET.SubElement(
            asset,
            "material",
            name=name,
            texture=name,
            specular=".08",
            shininess=".08",
            texrepeat=repeat,
            texuniform=uniform,
        )

    for name in ("hall", "oak", "sage", "slate", "rose"):
        mat("finish_" + name, finish(name, colors[name]), ".43 .43", "true")
    mat("plaster", finish("plaster", "dfdbcf"), ".70 .70", "true")
    for i in range(4):
        mat("art" + str(i), print_texture(i))
    mat("textile", finish("runner", colors["runner"]))
    mat("joinery", furniture_atlas())
    return files
