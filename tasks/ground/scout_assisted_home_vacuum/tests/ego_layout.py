"""Generate continuous furnished floor plans and unambiguous, grounded work orders."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Room:
    id: int
    kind: str
    rect: tuple
    finish: str

    @property
    def center(self):
        x0, y0, x1, y1 = self.rect
        return (x0 + x1) / 2, (y0 + y1) / 2


@dataclass(frozen=True)
class Door:
    a: int
    b: int
    xy: tuple
    along_x: bool
    width: float
    head: float


@dataclass(frozen=True)
class Fixture:
    room: int
    mesh: str
    xyz: tuple
    size: tuple
    turn: int

    @property
    def half_xy(self):
        w, d = self.size[:2]
        return (d / 2, w / 2) if self.turn % 180 else (w / 2, d / 2)


SUITES = {
    "bedroom": ("bedDouble", "cabinetBedDrawer", "lampRoundFloor", "plantSmall1"),
    "living room": ("loungeSofaLong", "tableCoffee", "loungeChair", "bookcaseOpen"),
    "study": ("desk", "bookcaseOpen", "chairDesk", "plantSmall1"),
    "kitchen": ("kitchenSink", "kitchenStove", "kitchenFridge", "kitchenCabinetDrawer"),
    "dining room": ("tableRound", "chairCushion", "chairCushion", "bookcaseOpen"),
    "reading room": ("bookcaseOpen", "loungeChair", "lampRoundFloor", "tableCoffee"),
    "laundry": ("kitchenCabinetDrawer", "kitchenSink", "bookcaseOpen", "plantSmall1"),
    "foyer": ("cabinetBedDrawer", "loungeChair", "plantSmall1", "lampRoundFloor"),
    "sitting room": ("loungeChair", "tableCoffee", "plantSmall1", "lampRoundFloor"),
}
DIM = {
    "bedDouble": (1.63, 2.04, 0.67),
    "cabinetBedDrawer": (0.60, 0.54, 0.58),
    "lampRoundFloor": (0.39, 0.39, 1.49),
    "plantSmall1": (0.48, 0.49, 0.72),
    "loungeSofaLong": (2.10, 0.94, 0.87),
    "tableCoffee": (0.98, 0.64, 0.43),
    "loungeChair": (0.85, 0.83, 0.88),
    "bookcaseOpen": (1.06, 0.43, 1.58),
    "desk": (1.50, 0.76, 0.79),
    "chairDesk": (0.57, 0.59, 0.84),
    "kitchenSink": (1.28, 0.63, 0.87),
    "kitchenStove": (0.80, 0.67, 0.88),
    "kitchenFridge": (0.74, 0.77, 1.75),
    "kitchenCabinetDrawer": (1.02, 0.65, 0.88),
    "tableRound": (1.05, 1.05, 0.76),
    "chairCushion": (0.56, 0.56, 0.85),
}
SIDES = ("north", "east", "south", "west")


def shift(p, v, d):
    return p[0] + v[0] * d, p[1] + v[1] * d


class Apartment:
    def __init__(self, seed, order_key=None):
        r = random.Random(seed)
        self.seed = seed
        cols, rows = r.choice((3, 4)), 3
        self.cols, self.rows = cols, rows
        xs, ys = [0.0], [0.0]
        for edge, n in ((xs, cols), (ys, rows)):
            for _ in range(n):
                edge.append(edge[-1] + r.uniform(6.3, 8.0))
        self.width, self.depth = xs[-1], ys[-1]
        n = cols * rows
        neighbors = []
        for i in range(n):
            if i % cols < cols - 1:
                neighbors.append((i, i + 1))
            if i // cols < rows - 1:
                neighbors.append((i, i + cols))
        self.foyer = r.randrange(n)
        adjacent = [
            b if a == self.foyer else a for a, b in neighbors if self.foyer in (a, b)
        ]
        self.laundry = r.choice(adjacent)
        service = tuple(sorted((self.foyer, self.laundry)))
        rest = [e for e in neighbors if e != service]
        r.shuffle(rest)
        parent = list(range(n))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        edges, spare = [], []
        for a, b in [service, *rest]:
            u, v = root(a), root(b)
            if u != v:
                parent[u] = v
                edges.append((a, b))
            else:
                spare.append((a, b))
        r.shuffle(spare)
        edges.extend(spare[: r.randint(1, min(4, len(spare)))])
        doors = {}
        for a, b in edges:
            ix, iy = a % cols, a // cols
            horizontal = b - a == cols
            cx, cy = (xs[ix] + xs[ix + 1]) / 2, (ys[iy] + ys[iy + 1]) / 2
            off = r.uniform(-0.48, 0.48)
            pt = (cx + off, ys[iy + 1]) if horizontal else (xs[ix + 1], cy + off)
            doors[a, b] = Door(
                a, b, pt, horizontal, r.uniform(1.64, 2.14), r.uniform(1.96, 2.19)
            )
        self.doors = doors
        self.segments = []  # continuous wall centre lines, split at real doorways

        def wall(lo, hi, fixed, along_x, d=None, outer=False):
            intervals = [(lo, hi)]
            if d is not None:
                t = d.xy[0 if along_x else 1]
                intervals = [(lo, t - d.width / 2), (t + d.width / 2, hi)]
            for l, h in intervals:
                p, q = ((l, fixed), (h, fixed)) if along_x else ((fixed, l), (fixed, h))
                self.segments.append((p, q, outer))

        for i in range(n):
            j, k = i % cols, i // cols
            if j == 0:
                wall(ys[k], ys[k + 1], 0.0, False, outer=True)
            if k == 0:
                wall(xs[j], xs[j + 1], 0.0, True, outer=True)
            wall(
                ys[k],
                ys[k + 1],
                xs[j + 1],
                False,
                doors.get((i, i + 1)) if j < cols - 1 else None,
                j == cols - 1,
            )
            wall(
                xs[j],
                xs[j + 1],
                ys[k + 1],
                True,
                doors.get((i, i + cols)),
                k == rows - 1,
            )
        kinds = {self.foyer: "foyer", self.laundry: "laundry"}
        free = [i for i in range(n) if i not in kinds]
        south = r.choice([i for i in free if i // cols == 0])
        north = r.choice([i for i in free if i // cols == rows - 1])
        kinds[south] = kinds[north] = "bedroom"
        free = [i for i in free if i not in (south, north)]
        r.shuffle(free)
        for kind in ("living room", "study", "kitchen", "dining room", "reading room"):
            kinds[free.pop()] = kind
        for i in free:
            kinds[i] = "sitting room"
        self.rooms = []
        self.fixtures = []
        for i in range(n):
            j, k = i % cols, i // cols
            rect = (xs[j], ys[k], xs[j + 1], ys[k + 1])
            room = Room(
                i, kinds[i], rect, r.choice(("oak", "slate", "rose", "sage", "hall"))
            )
            self.rooms.append(room)
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            r.shuffle(corners)
            for mesh, (sx, sy) in zip(SUITES[room.kind], corners):
                size, turn = DIM[mesh], r.choice((0, 90, 180, 270))
                hx, hy = (
                    (size[1] / 2, size[0] / 2)
                    if turn % 180
                    else (size[0] / 2, size[1] / 2)
                )
                x = rect[0] + hx + 0.33 if sx < 0 else rect[2] - hx - 0.33
                y = rect[1] + hy + 0.33 if sy < 0 else rect[3] - hy - 0.33
                self.fixtures.append(Fixture(i, mesh, (x, y, 0.0), size, turn))
        self.bedrooms = (south, north)
        self.living = next(p.id for p in self.rooms if p.kind == "living room")
        self.study = next(p.id for p in self.rooms if p.kind == "study")
        self.shelf = next(
            f
            for f in self.fixtures
            if f.room == self.study and f.mesh == "bookcaseOpen"
        )
        gate = doors[service]
        fx, fy = self.rooms[self.foyer].center
        lx, ly = self.rooms[self.laundry].center
        self.axis = (
            (0, 1 if ly > fy else -1) if gate.along_x else (1 if lx > fx else -1, 0)
        )
        self.stage = shift(gate.xy, self.axis, 1.20)
        # Both end positions leave a real turning bay behind the vacuum. A parked
        # carton must not close the sole doorway of a leaf room.
        self.park = {
            color: shift(gate.xy, self.axis, -d)
            for color, d in (("blue", 1.75), ("amber", 3.05))
        }
        self.box = shift(gate.xy, self.axis, -1.12)
        side = (-self.axis[1], self.axis[0])
        self.dock = shift(shift(gate.xy, self.axis, -2.72), side, 0.91)
        self.air_start = shift(self.dock, side, -0.86)
        self.spots = {}
        for i in (south, north, self.living):
            room = self.rooms[i]
            x0, y0, x1, y1 = room.rect
            cx, cy = room.center
            for side_name in SIDES:
                if side_name in ("north", "south"):
                    y = y1 - 0.90 if side_name == "north" else y0 + 0.90
                    points = [(cx - 0.85, y), (cx + 0.85, y)]
                else:
                    x = x1 - 0.90 if side_name == "east" else x0 + 0.90
                    points = [(x, cy - 0.85), (x, cy + 0.85)]
                for j, p in enumerate(points):
                    self.spots[i, side_name, j] = p
        self.spots["under-carton"] = self.box
        wall_pairs = (("north", "south"), ("east", "west"))
        bedroom_axis, living_axis = r.choice(wall_pairs), r.choice(wall_pairs)
        order = random.Random(r.getrandbits(96) if order_key is None else order_key)
        pole = order.randrange(2)
        self.requested_bedroom = self.bedrooms[pole]
        self.bed_wall, self.living_wall = (
            order.choice(bedroom_axis),
            order.choice(living_axis),
        )
        self.parking = order.choice(("blue", "amber"))
        # Only two wall pairs per room contain powder. Which is dirt depends on the order.
        self.spots = {
            k: v
            for k, v in self.spots.items()
            if k == "under-carton"
            or k[1] in (living_axis if k[0] == self.living else bedroom_axis)
        }
        self.wanted = {(self.requested_bedroom, self.bed_wall, j) for j in range(2)}
        self.wanted.update((self.living, self.living_wall, j) for j in range(2))
        self.wanted.add("under-carton")
        self.rug = self.rooms[
            next(i for i in range(n) if kinds[i] == "dining room")
        ].center
        self.prompt = (
            f"Clean both spills beside the {self.bed_wall} wall of the "
            f"{'northern' if pole else 'southern'} bedroom (there are two bedrooms), "
            f"both beside the {self.living_wall} wall in the living room, and the spill under "
            "the supply carton. Leave all other powder and the woven rug untouched. "
            "The job slip on top of the study bookcase shows the final parking mat. "
            "Push the carton onto the linen mat just inside the laundry, then back to the "
            "indicated foyer mat. Return the vacuum to the dock. Both robots start facing north."
        )

    def room_at(self, xy):
        x, y = xy
        return next(
            (
                r.id
                for r in self.rooms
                if r.rect[0] < x < r.rect[2] and r.rect[1] < y < r.rect[3]
            ),
            None,
        )

    def near_wall(self, p, clearance):
        x, y = p
        for (x0, y0), (x1, y1), _ in self.segments:
            if (
                math.hypot(x - min(max(x, x0), x1), y - min(max(y, y0), y1))
                < clearance + 0.09
            ):
                return True
        return False
