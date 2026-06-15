"""Adaptive label placement engine (brief: label_placement_engine).

Pure geometry — no PIL/torch/ComfyUI. The compositor measures label strip sizes
(via text_render) and asks this engine where to put each label so that, across a
dense frame, labels avoid leaving the image, avoid overlapping each other, avoid
covering primary subjects, and stay temporally stable (a track keeps its relative
label side between frames unless it no longer fits).

Flow per frame:
    placer = LabelPlacer(W, H, ...)   # persists across frames (temporal memory)
    placed = placer.place([PlacementRequest(...), ...])   # priority-ordered
Each Placed gives the resolved rect (clamped, in-bounds), the chosen position key,
and a ``visible`` flag (False = hide text under collision policy, keep the frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import geometry

Box = tuple

# Candidate positions, in default preference order (above first — editorial look).
POSITIONS = (
    "above_left", "above_right",
    "below_left", "below_right",
    "inside_top_left", "inside_top_right",
    "right_side", "left_side",
)

COLLISION_MODES = (
    "none", "hide_low_priority", "reposition", "reposition_and_shorten", "dense_editorial",
)


@dataclass
class PlacementRequest:
    key: str                       # track id (temporal memory key)
    anchor: tuple                  # detection box xyxy
    tw: int                        # label strip width (px)
    th: int                        # label strip height (px)
    priority: int = 50
    is_primary: bool = False


@dataclass
class Placed:
    key: str
    rect: tuple                    # resolved label rect xyxy (clamped in-bounds)
    position: str
    visible: bool = True


def candidate_rect(anchor, tw: int, th: int, position: str, gap: int = 4) -> tuple:
    """Unclamped candidate rect for a label of size (tw,th) at the given position
    relative to ``anchor`` (xyxy)."""
    x1, y1, x2, y2 = anchor
    if position == "above_left":
        ox, oy = x1, y1 - th - gap
    elif position == "above_right":
        ox, oy = x2 - tw, y1 - th - gap
    elif position == "below_left":
        ox, oy = x1, y2 + gap
    elif position == "below_right":
        ox, oy = x2 - tw, y2 + gap
    elif position == "inside_top_left":
        ox, oy = x1 + gap, y1 + gap
    elif position == "inside_top_right":
        ox, oy = x2 - tw - gap, y1 + gap
    elif position == "right_side":
        ox, oy = x2 + gap, y1
    elif position == "left_side":
        ox, oy = x1 - tw - gap, y1
    else:
        ox, oy = x1, y1 - th - gap
    return (float(ox), float(oy), float(ox + tw), float(oy + th))


def _overlap_area(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _out_of_bounds_area(rect, W, H) -> float:
    total = max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1]))
    inb = _overlap_area(rect, (0, 0, W, H))
    return total - inb


class LabelPlacer:
    """Stateful across frames: remembers each key's last position for stability."""

    def __init__(self, W: int, H: int, *, collision_mode: str = "reposition_and_shorten",
                 stickiness: float = 0.6, gap: int = 4):
        self.W = int(W)
        self.H = int(H)
        self.collision_mode = collision_mode if collision_mode in COLLISION_MODES else "reposition_and_shorten"
        self.stickiness = float(stickiness)
        self.gap = int(gap)
        self._last_pos: dict = {}     # key -> position str (temporal memory)

    # -- cost of placing `rect` (at `position`) for request `req` given placed list --
    def _cost(self, req: PlacementRequest, rect, position, placed, anchors):
        W, H = self.W, self.H
        diag = (W * W + H * H) ** 0.5
        cost = 0.0
        # boundary overflow (heavy)
        oob = _out_of_bounds_area(rect, W, H)
        cost += 6.0 * (oob / max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1])))
        # overlap with already-placed labels (heavier for higher-priority neighbours)
        for p in placed:
            ov = _overlap_area(rect, p.rect)
            if ov > 0:
                strength = 4.0
                cost += strength * (ov / max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1])))
        # covering the center of a primary subject
        for a, is_primary in anchors:
            if a is req.anchor:
                continue
            if is_primary:
                cx, cy = geometry.centroid(a) if hasattr(geometry, "centroid") else (
                    (a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
                if rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]:
                    cost += 2.0
        # connector length (distance from anchor edge) — prefer hugging the box
        ax, ay = (req.anchor[0] + req.anchor[2]) / 2, (req.anchor[1] + req.anchor[3]) / 2
        rx, ry = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        cost += 0.4 * (((ax - rx) ** 2 + (ay - ry) ** 2) ** 0.5 / max(1.0, diag))
        # temporal jump: penalty for changing from last frame's position
        last = self._last_pos.get(req.key)
        if last is not None and last != position:
            cost += self.stickiness
        # mild ordering preference (earlier positions slightly cheaper)
        cost += 0.02 * POSITIONS.index(position)
        return cost

    def place(self, requests) -> list:
        """Place labels (priority desc). Returns list[Placed] in the SAME order as input."""
        anchors = [(r.anchor, r.is_primary) for r in requests]
        order = sorted(range(len(requests)), key=lambda i: (-requests[i].priority, i))
        placed: list = []
        result: dict = {}
        for i in order:
            req = requests[i]
            best = None
            for pos in POSITIONS:
                rect = candidate_rect(req.anchor, req.tw, req.th, pos, self.gap)
                c = self._cost(req, rect, pos, placed, anchors)
                if best is None or c < best[0]:
                    best = (c, rect, pos)
            cost, rect, pos = best
            # clamp into frame for final rendering
            rect = geometry.clamp_box(rect, self.W, self.H) if hasattr(geometry, "clamp_box") else rect
            visible = True
            # collision policy: decide whether to hide this label's text
            heavy_overlap = any(_overlap_area(rect, p.rect) > 0.5 * max(1.0, req.tw * req.th)
                                for p in placed)
            if self.collision_mode == "none":
                visible = True
            elif self.collision_mode == "hide_low_priority" and heavy_overlap and not req.is_primary:
                visible = False
            elif self.collision_mode in ("reposition", "reposition_and_shorten") and heavy_overlap and not req.is_primary and cost > 3.0:
                visible = False
            # dense_editorial keeps everything visible (overlap accepted as style)
            p = Placed(key=req.key, rect=rect, position=pos, visible=visible)
            result[i] = p
            if visible:
                placed.append(p)
            self._last_pos[req.key] = pos
        return [result[i] for i in range(len(requests))]


def needs_shorten(placed: Placed, req: PlacementRequest, W: int, H: int) -> bool:
    """True if the resolved rect was clipped by the boundary (caller should try a
    shorter text tier before giving up)."""
    return _out_of_bounds_area(candidate_rect(req.anchor, req.tw, req.th, placed.position), W, H) > 1.0


__all__ = ["POSITIONS", "COLLISION_MODES", "PlacementRequest", "Placed",
           "LabelPlacer", "candidate_rect", "needs_shorten"]
