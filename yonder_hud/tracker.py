"""tracker.py — identity unification, smoothing and lifecycle (brief §10, §11).

This module turns a per-frame stream of :class:`~yonder_hud.types.Detection`
objects into stable, smoothed :class:`~yonder_hud.types.Track` identities and a
deterministic visual style for each one. It is the seam that makes the HUD look
*temporally coherent* (the same subject keeps the same id / colour / primitive
across frames) and *seed-stable* (a given ``(seed, track_id)`` always produces the
same style on every machine and run).

Two identity sources, in priority order (see ``docs/runtime_findings.md`` §3.5):

1. **PASSTHROUGH** — when detections already carry a ``source_track_id`` (native
   persistent SAM3 track data), the track identity is taken verbatim from that id.
   No IoU/centroid matching is performed; native index *is* the identity and it
   survives short occlusions for free. Each distinct ``source_track_id`` maps to a
   stable internal ``Track`` whose ``track_id`` equals that source id.
2. **GREEDY IoU + centroid + area** — when detections have no ``source_track_id``
   (the BOUNDING_BOX / MASK fallback path), live tracks are matched to this
   frame's detections by a deterministic combined score. Unmatched detections spawn
   new tracks with monotonically increasing ids that are **never reused**.

Smoothing is an exponential moving average over the matched box
(:func:`yonder_hud.geometry.ema_box`), optionally motion-adaptive
(:func:`yonder_hud.geometry.adaptive_alpha`). Tracks that stop being seen *age* and
are killed after ``memory_frames`` frames of absence (TTL lifecycle).

Style assignment (:meth:`Tracker.assign_style`) is derived once per track from
``stable_hash(seed, track_id)`` and then cached, so a track's colour / primitive /
asset / probability gates never flicker between frames. ``style_mode`` can force a
single primitive (``'corners'``/``'outline'``/``'solid'``/``'target_only'``/
``'labels_only'``/``'assets_only'``/``'procedural_only'``) or leave the full
``'mixed'`` visual language.

The module is schema-agnostic and Comfy-Cloud-safe: it imports only the Python
standard library, :mod:`yonder_hud.types` and the pure sibling helpers
:mod:`yonder_hud.geometry`, :mod:`yonder_hud.palette` and :mod:`yonder_hud.utils`.
No ComfyUI imports, no torch/PIL needed, no I/O, no networking, no subprocess, no
``eval``/``exec`` — pure deterministic compute.

See ``docs/INTERFACES.md`` ("## tracker.py") for the binding signatures.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

from . import geometry
from . import palette as palette_mod
from .types import (
    PRIMITIVE_STYLES,
    Box,
    Detection,
    Track,
    TrackStyle,
)
from .utils import stable_hash, unit_float

__all__ = [
    "Tracker",
    "micro_variation",
]


# Identity keys may be ints (greedy counter, or native int index) or strings
# (a native string source_track_id). Both flow through stable_hash type-safely.
TrackId = Union[int, str]


# ---------------------------------------------------------------------------
# Internal scalar helpers (kept private — not part of the public contract).
# ---------------------------------------------------------------------------

def _finite(value, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, mapping ``None``/NaN/Inf to ``default``."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    """Clamp a float into the closed unit interval ``[0.0, 1.0]`` (NaN -> 0.0)."""
    if value != value:  # NaN
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _det_box(det: Detection) -> Box:
    """Return a Detection's box as a canonical (normalized) ``xyxy`` tuple.

    The adapters already guarantee ``x2 >= x1`` / ``y2 >= y1``, but we normalize
    defensively so a hand-built fixture or a degenerate native bbox can never feed
    a reversed box into the geometry/EMA math.
    """
    return geometry.normalize_box(det.x1, det.y1, det.x2, det.y2)


# ---------------------------------------------------------------------------
# style_mode -> forced primitive resolution
# ---------------------------------------------------------------------------
#
# A ``style_mode`` either forces every track onto a single primitive, or (for the
# mixed / asset / procedural family) lets the per-track hash choose. The map below
# is the single source of truth for "which modes force which primitive".

#: style_mode values that pin every track to one specific primitive.
_FORCED_PRIMITIVE: Dict[str, str] = {
    "corners": "corners",
    "outline": "outline",
    "solid": "solid",
    "target_only": "target_only",
    # 'labels_only' renders no box primitive at all; we still need a concrete
    # value on TrackStyle.primitive_style, so we pin it to 'outline' but force the
    # caption on and every box-fill/target gate off (handled in assign_style).
    "labels_only": "outline",
    # 'assets_only' forces the user-PNG frame asset primitive (when one is
    # eligible); falls back gracefully when there are no assets (handled below).
    "assets_only": "frame_asset",
}

#: The procedural primitives the per-track hash may pick from in mixed mode. This
#: deliberately EXCLUDES ``'frame_asset'`` — whether a track uses a frame PNG is a
#: separate gate (``frame_asset_id``), so the box primitive stays a vector style.
_MIXED_PRIMITIVES: Tuple[str, ...] = ("corners", "outline", "solid", "target_only")


# Probability gates use *named* hash sub-streams so that, e.g., changing the
# display-probability threshold does not also reshuffle which tracks use a fill.
# Each gate draws an independent unit_float keyed by the stream name.
_STREAM_PRIMITIVE = "primitive"
_STREAM_FILL = "fill"
_STREAM_TARGET = "target"
_STREAM_SHOW = "show"
_STREAM_CAPTION = "caption"
_STREAM_CAPTION_SIDE = "caption_side"
_STREAM_GLITCH = "glitch"
_STREAM_FRAME_ASSET = "frame_asset"
_STREAM_TARGET_ASSET = "target_asset"


class Tracker:
    """Greedy/passthrough identity tracker with EMA smoothing and TTL lifecycle.

    A single :class:`Tracker` instance is created per render run and fed one frame
    at a time via :meth:`update`. It owns:

    * the live :class:`~yonder_hud.types.Track` table (keyed by ``track_id``),
    * a monotonically increasing id counter for the greedy path (ids are **never
      reused**, even after a track dies),
    * the per-track :class:`~yonder_hud.types.TrackStyle` cache.

    The constructor parameters mirror the user-facing node widgets (brief §23) and
    the binding signature in ``docs/INTERFACES.md``.

    Parameters
    ----------
    seed : int
        HUD seed. Drives every deterministic style / probability choice.
    iou_threshold : float
        Minimum IoU for the greedy matcher to even *consider* a detection a match
        for a live track. Detections below this against every track spawn new ids.
        (Tracks may still match on centroid+area above this floor.)
    max_centroid_ratio : float
        Maximum centroid displacement, as a fraction of the frame diagonal, for a
        greedy match to be allowed. A detection further than this from a track's
        last centroid can never match it (prevents teleporting identities).
    memory_frames : int
        Time-to-live: a track that is not matched for ``memory_frames`` consecutive
        frames is removed. ``0`` kills a track the first frame it is missing.
    smoothing : float
        UI smoothing in ``0..1`` (mapped to an EMA alpha via
        :func:`yonder_hud.geometry.smoothing_to_alpha`). ``0`` = raw, ``~0.9`` =
        very smooth/laggy. Applies on BOTH the passthrough and greedy paths.
    palette : sequence
        Resolved palette (list of ``(r,g,b)`` 0..1), e.g. from
        :func:`yonder_hud.palette.resolve_palette`. Used by
        :func:`yonder_hud.palette.color_for_track` to pick a stable ``color_id``.
    n_frame_assets : int
        Number of eligible (connected) frame-PNG assets. ``frame_asset_id`` is in
        ``[0, n_frame_assets)`` or ``-1`` when none / no assets.
    n_target_assets : int
        Number of eligible target-PNG assets (``target_asset_id`` analogously).
    style_mode : str
        One of :data:`yonder_hud.types.STYLE_MODES`. Forced modes pin every track
        to a single primitive; ``'mixed'`` lets the hash choose.
    display_probability : float
        Probability (per track, stable) that a track is shown at all. A track with
        ``show=False`` is suppressed every frame (deterministic flicker of clutter,
        brief §11.1). Clamped to ``0..1``.
    fill_probability : float
        Per-track probability the track uses a translucent fill. Clamped to
        ``0..1``. Ignored when ``style_mode`` forces a non-fill primitive.
    target_probability : float
        Per-track probability the track shows a crosshair/target. Clamped to
        ``0..1``. Ignored when ``style_mode`` forces a non-target primitive.
    W, H : int
        Frame width/height in pixels (used to compute the frame diagonal for
        centroid gating and adaptive smoothing).
    adaptive : bool
        When ``True``, the EMA alpha is motion-adaptive
        (:func:`yonder_hud.geometry.adaptive_alpha`): fast motion snaps, slow
        motion is smoothed hard. When ``False``, a constant alpha is used.
    """

    def __init__(
        self,
        *,
        seed: int,
        iou_threshold: float,
        max_centroid_ratio: float,
        memory_frames: int,
        smoothing: float,
        palette,
        n_frame_assets: int,
        n_target_assets: int,
        style_mode: str,
        display_probability: float,
        fill_probability: float = 0.25,
        target_probability: float = 0.7,
        W: int,
        H: int,
        adaptive: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.iou_threshold = _finite(iou_threshold, 0.0)
        self.max_centroid_ratio = _finite(max_centroid_ratio, 0.35)
        # memory_frames is a count; clamp to >= 0 (a negative TTL would never keep
        # a track). int() is defensive against float widgets.
        self.memory_frames = max(0, int(memory_frames))
        self.smoothing = _clamp01(_finite(smoothing, 0.0))

        # Palette must be non-empty for color_for_track; fall back to a single
        # white if a caller passes something empty/None (color is still defined).
        if palette is None or len(palette) == 0:
            self.palette = [(1.0, 1.0, 1.0)]
        else:
            self.palette = list(palette)

        self.n_frame_assets = max(0, int(n_frame_assets))
        self.n_target_assets = max(0, int(n_target_assets))

        self.style_mode = style_mode if isinstance(style_mode, str) else "mixed"
        self.display_probability = _clamp01(_finite(display_probability, 1.0))
        self.fill_probability = _clamp01(_finite(fill_probability, 0.25))
        self.target_probability = _clamp01(_finite(target_probability, 0.7))

        self.W = max(0, int(W))
        self.H = max(0, int(H))
        # Frame diagonal in pixels (>= 0). Zero-sized frames degrade gracefully:
        # centroid gating is then disabled (every distance is "within" the frame).
        self._frame_diag = math.hypot(float(self.W), float(self.H))

        self.adaptive = bool(adaptive)

        # The EMA alpha for the constant-smoothing case (recomputed once here).
        self._base_alpha = geometry.smoothing_to_alpha(self.smoothing)

        # Live track table, keyed by track_id (int or str).
        self._tracks: Dict[TrackId, Track] = {}

        # Monotonic id counter for the greedy path. Starts at 1 so ids read like
        # human "object 1, object 2, …". NEVER decremented; ids are never reused.
        self._next_id: int = 1

        # Per-track style cache. Styles are assigned once and memoised so a track's
        # look never changes between frames.
        self._style_cache: Dict[TrackId, TrackStyle] = {}

    # ------------------------------------------------------------------
    # Public read-only state
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> Dict[TrackId, Track]:
        """The live track table, keyed by ``track_id``.

        Returns the internal dict (not a copy) so callers can inspect track state
        between frames. Treat it as read-only; mutating it corrupts the tracker.
        """
        return self._tracks

    # ------------------------------------------------------------------
    # Style assignment (deterministic, cached once per track)
    # ------------------------------------------------------------------

    def assign_style(self, track_id: TrackId) -> TrackStyle:
        """Return the deterministic :class:`~yonder_hud.types.TrackStyle` for a track.

        The style is derived purely from ``stable_hash(self.seed, track_id, <stream>)``
        for several independent named sub-streams, so it is identical on every run /
        machine and stable across frames. The result is cached on first call and
        the same object is returned thereafter (the contract: "assigned ONCE per
        track").

        Fields, and how they are chosen:

        * ``color_id`` — :func:`yonder_hud.palette.color_for_track` index.
        * ``primitive_style`` — forced by ``style_mode`` (see
          :data:`_FORCED_PRIMITIVE`) or, in ``'mixed'``/asset/procedural modes,
          picked from :data:`_MIXED_PRIMITIVES` by hash. ``'assets_only'`` uses a
          frame asset when one is eligible, else degrades to a vector primitive.
        * ``frame_asset_id`` — in ``[0, n_frame_assets)`` (hash-picked) or ``-1``
          when there are no eligible frame assets / the mode does not draw a frame
          asset.
        * ``target_asset_id`` — analogous, in ``[0, n_target_assets)`` or ``-1``.
        * ``show`` — ``unit_float(...) < display_probability``.
        * ``use_fill`` — ``unit_float(...) < fill_probability`` (unless the mode
          forces it on for ``'solid'`` or off for non-fill forced modes).
        * ``use_target`` — ``unit_float(...) < target_probability`` (forced on for
          ``'target_only'``, off for box-only forced modes / ``'labels_only'``).
        * ``use_caption`` — gated by ``style_mode`` (always on for ``'labels_only'``;
          on by default elsewhere, with a small deterministic suppression so not
          every object is labelled in ``'mixed'``).
        * ``caption_side`` — ``'top'`` / ``'bottom'`` by hash.
        * ``glitch_phase`` — a stable ``0..1`` personality offset for outline glitch.

        Args:
            track_id: The track identity (``int`` or ``str``).

        Returns:
            The cached :class:`~yonder_hud.types.TrackStyle` for ``track_id``.
        """
        cached = self._style_cache.get(track_id)
        if cached is not None:
            return cached

        seed = self.seed
        mode = self.style_mode

        # --- colour (stable palette index) ---------------------------------
        color_id, _rgb = palette_mod.color_for_track(self.palette, seed, track_id)

        # --- frame / target asset ids (hash-picked, within counts or -1) ---
        if self.n_frame_assets > 0:
            frame_asset_id = (
                stable_hash(seed, track_id, _STREAM_FRAME_ASSET) % self.n_frame_assets
            )
        else:
            frame_asset_id = -1
        if self.n_target_assets > 0:
            target_asset_id = (
                stable_hash(seed, track_id, _STREAM_TARGET_ASSET) % self.n_target_assets
            )
        else:
            target_asset_id = -1

        # --- primitive style (respect forced style_mode) ------------------
        forced = _FORCED_PRIMITIVE.get(mode)
        if forced is not None:
            primitive = forced
            # 'assets_only' wants a frame PNG; if none is eligible, degrade to a
            # deterministic vector primitive so the track is still drawn.
            if mode == "assets_only" and frame_asset_id < 0:
                primitive = _MIXED_PRIMITIVES[
                    stable_hash(seed, track_id, _STREAM_PRIMITIVE)
                    % len(_MIXED_PRIMITIVES)
                ]
        else:
            # mixed / procedural_only: hash-pick a vector primitive.
            primitive = _MIXED_PRIMITIVES[
                stable_hash(seed, track_id, _STREAM_PRIMITIVE) % len(_MIXED_PRIMITIVES)
            ]

        # In 'procedural_only' a frame asset must never be drawn (procedural means
        # vector primitives only); zero out the frame asset slot.
        if mode == "procedural_only":
            frame_asset_id = -1

        # In 'mixed', a track only actually *uses* its eligible frame asset when its
        # primitive rolled 'frame_asset'. Our mixed primitives don't include
        # 'frame_asset', so a frame asset is layered behind the vector primitive via
        # frame_asset_id remaining set; the compositor decides. We keep frame_asset_id
        # populated (eligible) so the renderer can choose to draw it. For forced
        # vector-only modes ('corners'/'outline'/'solid'/'target_only'/'labels_only')
        # the user explicitly asked for that primitive, so suppress frame assets.
        if mode in ("corners", "outline", "solid", "target_only", "labels_only"):
            frame_asset_id = -1

        # --- probability gates (independent named streams) ----------------
        show = unit_float(seed, track_id, _STREAM_SHOW) < self.display_probability

        if mode == "solid":
            use_fill = True
        elif mode in ("corners", "outline", "target_only", "labels_only"):
            use_fill = False
        else:  # mixed / assets_only / procedural_only
            use_fill = unit_float(seed, track_id, _STREAM_FILL) < self.fill_probability

        if mode == "target_only":
            use_target = True
        elif mode in ("corners", "outline", "solid", "labels_only"):
            use_target = False
        else:  # mixed / assets_only / procedural_only
            use_target = (
                unit_float(seed, track_id, _STREAM_TARGET) < self.target_probability
            )

        # --- caption gate --------------------------------------------------
        if mode == "labels_only":
            use_caption = True
        else:
            # Deterministically suppress captions on a minority of tracks so a busy
            # frame is not wall-to-wall text. Keep most labelled (threshold 0.85).
            use_caption = unit_float(seed, track_id, _STREAM_CAPTION) < 0.85

        # --- caption side & glitch personality ----------------------------
        caption_side = (
            "bottom"
            if (stable_hash(seed, track_id, _STREAM_CAPTION_SIDE) & 1)
            else "top"
        )
        glitch_phase = unit_float(seed, track_id, _STREAM_GLITCH)

        style = TrackStyle(
            color_id=int(color_id),
            primitive_style=primitive,
            frame_asset_id=int(frame_asset_id),
            target_asset_id=int(target_asset_id),
            show=bool(show),
            use_fill=bool(use_fill),
            use_target=bool(use_target),
            use_caption=bool(use_caption),
            caption_side=caption_side,
            glitch_phase=float(glitch_phase),
        )
        # Defensive: keep primitive_style inside the declared vocabulary.
        if style.primitive_style not in PRIMITIVE_STYLES:
            style.primitive_style = "corners"

        self._style_cache[track_id] = style
        return style

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self, frame_index: int, detections: List[Detection]
    ) -> List[Tuple[Track, Detection]]:
        """Advance the tracker by one frame and return this frame's visible pairs.

        Behaviour depends on whether the detections carry native identity:

        * **PASSTHROUGH** — if *any* detection has a non-``None``
          ``source_track_id``, the whole frame is treated as native-tracked: each
          such detection's identity is its ``source_track_id`` (no IoU). A detection
          without a ``source_track_id`` in an otherwise-native frame is given a
          deterministic synthetic id (``"_anon_<source_index>"``) so it is still
          tracked rather than dropped.
        * **GREEDY** — otherwise, live tracks are matched to detections by a
          combined IoU + centroid + area score, in descending score order, each
          track/detection used at most once. A match additionally requires
          ``IoU >= iou_threshold`` *or* a centroid within ``max_centroid_ratio`` of
          the frame diagonal (so a fast jump still re-acquires the same id), and is
          always rejected past the centroid gate. Unmatched detections create new
          tracks with fresh monotonic ids.

        In both paths each surviving/created track's ``smoothed_box`` is updated by
        an EMA of its raw detection box (constant or motion-adaptive alpha), its
        ``last_box`` / ``last_seen_frame`` / ``age`` are advanced and
        ``missing_frames`` reset to 0. Tracks that were not seen this frame have
        ``missing_frames`` incremented and are removed once that exceeds
        ``memory_frames`` (TTL). A track's :class:`~yonder_hud.types.TrackStyle` is
        assigned once on first sight via :meth:`assign_style`.

        Args:
            frame_index: The 0-based index of the frame being processed.
            detections: This frame's detections (xyxy, from the adapters). May be
                empty or ``None``.

        Returns:
            A list of ``(Track, Detection)`` pairs for the tracks visible **this**
            frame, sorted deterministically by ``track_id`` (ints before strings,
            each group ascending — see :func:`_track_id_sort_key`).
        """
        frame_index = int(frame_index)
        dets = list(detections) if detections else []

        # A frame is "native" iff at least one detection carries a source_track_id.
        native = any(getattr(d, "source_track_id", None) is not None for d in dets)

        if native:
            matched = self._update_passthrough(frame_index, dets)
        else:
            matched = self._update_greedy(frame_index, dets)

        # Age + cull tracks that were not matched this frame (TTL lifecycle).
        self._age_and_cull(frame_index, matched_ids={tid for tid, _ in matched})

        # Build (Track, Detection) result for the tracks seen this frame, ordered
        # deterministically by track_id.
        result: List[Tuple[Track, Detection]] = []
        for tid, det in matched:
            track = self._tracks.get(tid)
            if track is not None:
                result.append((track, det))
        result.sort(key=lambda pair: _track_id_sort_key(pair[0].track_id))
        return result

    # ------------------------------------------------------------------
    # PASSTHROUGH path (native source_track_id identity)
    # ------------------------------------------------------------------

    def _update_passthrough(
        self, frame_index: int, dets: List[Detection]
    ) -> List[Tuple[TrackId, Detection]]:
        """Map detections to tracks by their ``source_track_id`` (no IoU).

        When two detections in the same frame somehow claim the *same* identity
        (should not happen for real native data, where index is unique per frame),
        the larger-area detection wins for that id — deterministic and avoids a
        double-update of one track.
        """
        # Resolve each detection to a stable identity key. A native detection may,
        # in principle, have a missing source_track_id (mixed input); give it a
        # deterministic synthetic id so it is tracked rather than silently dropped.
        chosen: Dict[TrackId, Detection] = {}
        for det in dets:
            box = _det_box(det)
            if geometry.is_degenerate(box, min_size=1.0):
                continue
            sid = getattr(det, "source_track_id", None)
            if sid is None:
                tid: TrackId = "_anon_%d" % int(getattr(det, "source_index", 0) or 0)
            elif isinstance(sid, bool):
                # bool is a subclass of int; normalise so True/False don't masquerade
                # as the ints 1/0 elsewhere.
                tid = "_bool_%s" % ("1" if sid else "0")
            elif isinstance(sid, (int, str)):
                tid = sid
            else:
                tid = str(sid)

            prev = chosen.get(tid)
            if prev is None or geometry.area(box) > geometry.area(_det_box(prev)):
                chosen[tid] = det

        out: List[Tuple[TrackId, Detection]] = []
        for tid, det in chosen.items():
            self._touch_track(tid, frame_index, det)
            out.append((tid, det))
        return out

    # ------------------------------------------------------------------
    # GREEDY path (IoU + centroid + area)
    # ------------------------------------------------------------------

    def _update_greedy(
        self, frame_index: int, dets: List[Detection]
    ) -> List[Tuple[TrackId, Detection]]:
        """Greedy deterministic matching of live tracks to this frame's detections.

        The candidate score for a (track, detection) pair is::

            score = 0.6*IoU + 0.3*(1 - centroid_dist/frame_diag) + 0.1*area_sim

        with ``area_sim = min(a,b)/max(a,b)`` of the two box areas. A pair is only a
        *candidate* when it clears the gate (``IoU >= iou_threshold`` OR centroid
        within ``max_centroid_ratio`` of the frame diagonal) **and** never when the
        centroid is beyond that ratio. Candidates are consumed in descending score,
        ties broken deterministically by ``(track_id, source_index)``, each track
        and detection used at most once. Unmatched detections spawn new tracks.
        """
        # Pre-filter to drawable detection boxes (drop degenerate). Keep stable
        # order by source_index for deterministic tie-breaking.
        live_ids = sorted(self._tracks.keys(), key=_track_id_sort_key)
        usable: List[Tuple[int, Detection, Box]] = []
        for di, det in enumerate(dets):
            box = _det_box(det)
            if geometry.is_degenerate(box, min_size=1.0):
                continue
            usable.append((di, det, box))

        diag = self._frame_diag
        max_dist = self.max_centroid_ratio * diag if diag > 0.0 else float("inf")

        # Build all viable candidate edges (track_id, det_local_index, score).
        candidates: List[Tuple[float, TrackId, int]] = []
        for tid in live_ids:
            track = self._tracks[tid]
            tbox = track.smoothed_box
            t_area = geometry.area(tbox)
            for local_i, (_di, _det, dbox) in enumerate(usable):
                cdist = geometry.centroid_dist(tbox, dbox)
                # Hard centroid gate: a detection too far away can NEVER be this
                # track (prevents identity teleporting across the frame).
                if diag > 0.0 and cdist > max_dist:
                    continue
                iou = geometry.iou(tbox, dbox)
                # A pair must clear either the IoU floor or the centroid gate to be
                # considered at all (the centroid gate already passed above when
                # diag>0, so this mainly rescues low-IoU-but-close re-acquisition).
                if iou < self.iou_threshold and not (diag > 0.0 and cdist <= max_dist):
                    continue
                d_area = geometry.area(dbox)
                if t_area <= 0.0 or d_area <= 0.0:
                    area_sim = 0.0
                else:
                    area_sim = min(t_area, d_area) / max(t_area, d_area)
                if diag > 0.0:
                    centroid_term = 1.0 - min(1.0, cdist / diag)
                else:
                    centroid_term = 1.0
                score = 0.6 * iou + 0.3 * centroid_term + 0.1 * area_sim
                candidates.append((score, tid, local_i))

        # Greedy assignment in descending score. Deterministic tie-break: higher
        # score first, then lower track_id, then lower source_index.
        candidates.sort(
            key=lambda c: (
                -c[0],
                _track_id_sort_key(c[1]),
                usable[c[2]][0],  # original source_index
            )
        )

        used_tracks: set = set()
        used_dets: set = set()
        matched: List[Tuple[TrackId, Detection]] = []
        for _score, tid, local_i in candidates:
            if tid in used_tracks or local_i in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(local_i)
            det = usable[local_i][1]
            self._touch_track(tid, frame_index, det)
            matched.append((tid, det))

        # Unmatched detections -> brand-new tracks (monotonic, never-reused ids).
        for local_i, (_di, det, _box) in enumerate(usable):
            if local_i in used_dets:
                continue
            new_id = self._next_id
            self._next_id += 1
            self._touch_track(new_id, frame_index, det)
            matched.append((new_id, det))

        return matched

    # ------------------------------------------------------------------
    # Track creation / update shared by both paths
    # ------------------------------------------------------------------

    def _touch_track(self, tid: TrackId, frame_index: int, det: Detection) -> None:
        """Create or update the track ``tid`` from detection ``det`` on this frame.

        Applies EMA smoothing to the track's box (seeding on first sight), advances
        the lifecycle fields and ensures a style is assigned exactly once. The
        detection's label (if any) is carried onto the track for caption fallback.
        """
        raw_box = _det_box(det)
        existing = self._tracks.get(tid)

        if existing is None:
            # First sight: EMA seeds on the raw box (ema_box with previous=None
            # returns current). Style assigned once, now.
            style = self.assign_style(tid)
            track = Track(
                track_id=tid,
                last_box=raw_box,
                smoothed_box=raw_box,
                last_seen_frame=frame_index,
                age=1,
                missing_frames=0,
                label=det.label or "",
                style=style,
            )
            self._tracks[tid] = track
            return

        # Subsequent sight: blend toward the new raw box.
        prev_smoothed = existing.smoothed_box
        if self.adaptive:
            alpha = geometry.adaptive_alpha(
                self._base_alpha, raw_box, prev_smoothed, self._frame_diag
            )
        else:
            alpha = self._base_alpha
        smoothed = geometry.ema_box(raw_box, prev_smoothed, alpha)

        existing.last_box = raw_box
        existing.smoothed_box = smoothed
        existing.last_seen_frame = frame_index
        existing.age += 1
        existing.missing_frames = 0
        if det.label:
            existing.label = det.label
        # Style stays fixed (assigned once); ensure it is present defensively.
        if existing.style is None:
            existing.style = self.assign_style(tid)

    # ------------------------------------------------------------------
    # Lifecycle (TTL)
    # ------------------------------------------------------------------

    def _age_and_cull(self, frame_index: int, matched_ids: set) -> None:
        """Increment ``missing_frames`` for unseen tracks and cull beyond TTL.

        A track not in ``matched_ids`` this frame ages by one missing frame. Once
        ``missing_frames > memory_frames`` it is removed from the live table (its id
        is retired forever — the monotonic counter never hands it out again). The
        style cache is intentionally **not** purged, so if a *passthrough* identity
        with the same id reappears later it keeps its original look; greedy ids are
        never reused so their cached styles are simply inert.
        """
        to_delete: List[TrackId] = []
        for tid, track in self._tracks.items():
            if tid in matched_ids:
                continue
            track.missing_frames += 1
            track.age += 1
            if track.missing_frames > self.memory_frames:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]


# ---------------------------------------------------------------------------
# Deterministic track-id ordering
# ---------------------------------------------------------------------------

def _track_id_sort_key(track_id: TrackId):
    """Total, deterministic ordering key for a track id that may be int or str.

    Python forbids comparing ``int`` with ``str``; track ids can be either (greedy
    counter ints, native int indices, or native string ids). We sort all ints
    before all strings, each group ascending, with a stable string form as the
    final tiebreak. This keeps :meth:`Tracker.update`'s output order fully
    deterministic regardless of id types present.
    """
    if isinstance(track_id, bool):
        # Shouldn't normally reach here (bools are remapped), but be total.
        return (0, int(track_id), "")
    if isinstance(track_id, int):
        return (0, track_id, "")
    return (1, 0, str(track_id))


# ---------------------------------------------------------------------------
# Per-frame micro-variation (brief §11.2)
# ---------------------------------------------------------------------------

def micro_variation(
    seed: int,
    track_id: int,
    frame_index: int,
    *,
    flicker_p: float,
    style_flash_p: float,
    jitter_px: int,
    opacity_jitter: float,
) -> dict:
    """Deterministic per-frame flicker / jitter for one track on one frame.

    This is the *temporal personality* layer (brief §11.2): tiny, seed-stable
    per-frame variations that make a static-looking HUD feel alive (a corner
    momentarily dropping out, the fill flashing, a 1-px jitter) without ever
    changing a track's underlying identity/style. Every value is derived from
    independent named hash sub-streams keyed by ``(seed, track_id, frame_index,
    <stream>)``, so the same inputs always yield the same dict on every machine and
    run, and neighbouring frames/tracks vary independently.

    Args:
        seed: The HUD seed.
        track_id: The track identity (``int`` or ``str`` both accepted).
        frame_index: The 0-based frame index.
        flicker_p: Probability the line/box momentarily drops out this frame
            (``line_dropout``). Clamped to ``0..1``; ``0`` disables.
        style_flash_p: Probability of a one-frame "flash" of the target and/or fill
            (``target_flash`` / ``fill_flash``). Clamped to ``0..1``; ``0`` disables.
            The target and fill flashes are drawn from independent sub-streams so
            they do not always co-fire.
        jitter_px: Maximum absolute position jitter in pixels per axis. The returned
            ``jitter`` is an ``(dx, dy)`` int pair, each independently uniform in
            ``[-jitter_px, +jitter_px]``. ``0`` (or negative) -> ``(0, 0)``.
        opacity_jitter: Maximum fractional opacity wobble. ``opacity_mul`` is
            ``1 + u`` with ``u`` uniform in ``[-opacity_jitter, +opacity_jitter]``,
            then clamped so it never goes ``<= 0`` (the overlay never inverts).
            Clamped to ``0..1``; ``0`` -> ``opacity_mul == 1.0``.

    Returns:
        dict: ``{'opacity_mul': float, 'line_dropout': bool, 'target_flash': bool,
        'fill_flash': bool, 'jitter': (int, int)}``.
    """
    flicker_p = _clamp01(_finite(flicker_p, 0.0))
    style_flash_p = _clamp01(_finite(style_flash_p, 0.0))
    opacity_jitter = _clamp01(_finite(opacity_jitter, 0.0))
    try:
        jitter_px = int(jitter_px)
    except (TypeError, ValueError):
        jitter_px = 0
    if jitter_px < 0:
        jitter_px = 0

    # --- line dropout (flicker) ---------------------------------------
    line_dropout = (
        unit_float(seed, track_id, frame_index, "flicker") < flicker_p
        if flicker_p > 0.0
        else False
    )

    # --- target / fill flash (independent sub-streams) ----------------
    if style_flash_p > 0.0:
        target_flash = unit_float(seed, track_id, frame_index, "target_flash") < style_flash_p
        fill_flash = unit_float(seed, track_id, frame_index, "fill_flash") < style_flash_p
    else:
        target_flash = False
        fill_flash = False

    # --- positional jitter (each axis independent, integer pixels) ----
    if jitter_px > 0:
        # Map a unit float in [0,1) to an integer in [-jitter_px, +jitter_px]
        # inclusive (2*jitter_px + 1 buckets) deterministically.
        span = 2 * jitter_px + 1
        ux = unit_float(seed, track_id, frame_index, "jitter_x")
        uy = unit_float(seed, track_id, frame_index, "jitter_y")
        dx = int(ux * span) - jitter_px
        dy = int(uy * span) - jitter_px
        # Guard the (vanishingly unlikely) u==~1.0 edge so we stay in range.
        if dx > jitter_px:
            dx = jitter_px
        if dy > jitter_px:
            dy = jitter_px
        jitter = (dx, dy)
    else:
        jitter = (0, 0)

    # --- opacity wobble -----------------------------------------------
    if opacity_jitter > 0.0:
        # u in [-opacity_jitter, +opacity_jitter]
        u = (unit_float(seed, track_id, frame_index, "opacity") * 2.0 - 1.0) * opacity_jitter
        opacity_mul = 1.0 + u
        # Never let the overlay vanish entirely or go negative.
        if opacity_mul < 0.05:
            opacity_mul = 0.05
        if opacity_mul > 2.0:
            opacity_mul = 2.0
    else:
        opacity_mul = 1.0

    return {
        "opacity_mul": float(opacity_mul),
        "line_dropout": bool(line_dropout),
        "target_flash": bool(target_flash),
        "fill_flash": bool(fill_flash),
        "jitter": (int(jitter[0]), int(jitter[1])),
    }
