"""YONDER_ANNOTATIONS — model-independent semantic annotation layer.

The tracking layer (SAM3) gives stable identity + masks but NO descriptive prose.
This module defines a versioned, schema-validated annotation structure that can
carry short/medium/detailed labels, actions, context, attributes and child regions
(face, footwear, held objects), attached to stable namespaced track ids
(``person:0``, ``face:0`` …). It is transport-agnostic: an ``AnnotationSet`` object
or a JSON string round-trip identically.

Hard rules (enforced here, relied on by the renderer):
  * The renderer NEVER invents detailed descriptions — they live here, sourced from
    manual JSON, a VLM workflow, a manifest, or safe generic fallbacks.
  * Everything normalizes to absolute-pixel xyxy internally.
  * Sensitive attributes (identity, ethnicity, religion, nationality, medical …) are
    stripped defensively, in case an upstream VLM returns them.

Pure module: stdlib only (json/re/dataclasses/math/copy). No torch/PIL/ComfyUI.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1
ANNOTATIONS_TYPE = "YONDER_ANNOTATIONS"

# Label "length" tiers, longest -> shortest, used for adaptive fallback.
LABEL_TIERS = ("detailed_label", "medium_label", "short_label")
CAPTION_MODES = ("auto", "short", "medium", "detailed", "action", "technical", "mixed")

# ---------------------------------------------------------------------------
# Sensitive-attribute defense (non_goals: no identity / protected attributes)
# ---------------------------------------------------------------------------
_SENSITIVE_WORDS = [
    # ethnicity / race / nationality
    "ethnicity", "race", "racial", "caucasian", "asian", "african", "hispanic",
    "latino", "latina", "arab", "jewish", "white man", "black man", "white woman",
    "black woman", "nationality",
    # religion
    "religion", "religious", "muslim", "christian", "hindu", "buddhist", "jew", "islamic",
    # medical / disability
    "disabled", "disability", "medical", "illness", "syndrome", "diagnosis", "patient",
    # identity
    "named", "name is", "celebrity", "famous", "identified as",
]
# Word-boundary anchored so e.g. "jew" never matches "jewelry" (an allowed accessory).
_SENSITIVE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _SENSITIVE_WORDS) + r")\b", re.IGNORECASE)


def filter_sensitive(text: Optional[str]) -> str:
    """Remove sensitive/protected words/phrases (identity, ethnicity, religion,
    nationality, medical) INLINE, keeping the rest of the description.

    e.g. "MUSLIM MAN WITH TATTOOS" -> "MAN WITH TATTOOS". Never raises.
    """
    if not text:
        return ""
    cleaned = _SENSITIVE_RE.sub(" ", str(text))
    cleaned = re.sub(r"\s+([,/])", r"\1", cleaned)   # tidy " ,"
    cleaned = re.sub(r"[,/]\s*(?=[,/])", "", cleaned)  # drop emptied segments
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,/")
    return cleaned


def _filter_attr_list(items) -> list:
    out = []
    for it in (items or []):
        f = filter_sensitive(str(it))
        if f:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0

    def to_dict(self):
        return {"width": self.width, "height": self.height, "fps": self.fps,
                "frame_count": self.frame_count}


@dataclass
class StyleOverrides:
    priority: int = 50
    preferred_color: Optional[str] = None
    preferred_asset: Optional[int] = None
    preferred_label_mode: Optional[str] = None

    def to_dict(self):
        return {"priority": self.priority, "preferred_color": self.preferred_color,
                "preferred_asset": self.preferred_asset,
                "preferred_label_mode": self.preferred_label_mode}


@dataclass
class ChildAnnotation:
    annotation_id: str
    category: str = "object"
    short_label: str = ""
    detailed_label: str = ""
    confidence: Optional[float] = None
    region_source: str = "secondary_detector"   # secondary_detector|vlm|manual|heuristic|track
    region_mode: str = "bbox"                    # bbox|mask|track_relative
    bbox_by_frame: dict = field(default_factory=dict)  # {frame_index:int -> [x1,y1,x2,y2] abs xyxy}
    parent_track_id: Optional[str] = None
    priority: int = 40

    def bbox_for(self, frame_index: int):
        """Return abs-xyxy for this frame, else the nearest earlier keyframe, else None."""
        if not self.bbox_by_frame:
            return None
        if frame_index in self.bbox_by_frame:
            return self.bbox_by_frame[frame_index]
        keys = sorted(self.bbox_by_frame)
        prior = [k for k in keys if k <= frame_index]
        return self.bbox_by_frame[prior[-1]] if prior else self.bbox_by_frame[keys[0]]

    def to_dict(self):
        return {"annotation_id": self.annotation_id, "category": self.category,
                "short_label": self.short_label, "detailed_label": self.detailed_label,
                "confidence": self.confidence, "region_source": self.region_source,
                "region_mode": self.region_mode,
                "bbox_by_frame": {str(k): v for k, v in self.bbox_by_frame.items()},
                "parent_track_id": self.parent_track_id, "priority": self.priority}


@dataclass
class TrackAnnotation:
    track_id: str                      # namespaced, e.g. "person:0"
    source: str = "sam3_primary"
    source_track_index: Optional[int] = None
    category: str = "object"
    short_label: str = ""
    medium_label: str = ""
    detailed_label: str = ""
    action_label: str = ""
    context_label: str = ""
    display_label: str = ""
    confidence: Optional[float] = None
    attributes: dict = field(default_factory=dict)
    style_overrides: StyleOverrides = field(default_factory=StyleOverrides)
    children: list = field(default_factory=list)   # list[ChildAnnotation]

    def to_dict(self):
        return {"track_id": self.track_id, "source": self.source,
                "source_track_index": self.source_track_index, "category": self.category,
                "short_label": self.short_label, "medium_label": self.medium_label,
                "detailed_label": self.detailed_label, "action_label": self.action_label,
                "context_label": self.context_label, "display_label": self.display_label,
                "confidence": self.confidence, "attributes": self.attributes,
                "style_overrides": self.style_overrides.to_dict(),
                "children": [c.to_dict() for c in self.children]}


@dataclass
class FrameAnnotation:
    annotation_id: str
    category: str = "object"
    short_label: str = ""
    detailed_label: str = ""
    bbox: Optional[list] = None        # [x1,y1,x2,y2] abs xyxy
    confidence: Optional[float] = None
    priority: int = 60
    parent_track_id: Optional[str] = None
    frame_index: int = 0

    def to_dict(self):
        return {"annotation_id": self.annotation_id, "parent_track_id": self.parent_track_id,
                "category": self.category, "short_label": self.short_label,
                "detailed_label": self.detailed_label, "bbox": self.bbox,
                "confidence": self.confidence, "priority": self.priority}


@dataclass
class AnnotationSet:
    version: int = SCHEMA_VERSION
    video: VideoMeta = field(default_factory=VideoMeta)
    tracks: list = field(default_factory=list)              # list[TrackAnnotation]
    frame_annotations: dict = field(default_factory=dict)   # {frame_index:int -> [FrameAnnotation]}

    def tracks_by_id(self) -> dict:
        return {t.track_id: t for t in self.tracks}

    def get_track(self, track_id: str):
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def frame_list(self, frame_index: int) -> list:
        return self.frame_annotations.get(frame_index, [])

    def is_empty(self) -> bool:
        return not self.tracks and not self.frame_annotations

    def to_dict(self):
        return {"version": self.version, "video": self.video.to_dict(),
                "tracks": [t.to_dict() for t in self.tracks],
                "frame_annotations": {str(k): [a.to_dict() for a in v]
                                      for k, v in self.frame_annotations.items()}}

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


# ---------------------------------------------------------------------------
# Coordinate coercion (universal detection adapter rules)
# ---------------------------------------------------------------------------

def coerce_bbox(raw, W: int, H: int, fmt: str = "auto") -> Optional[list]:
    """Normalize a bbox in any of {xyxy,xywh}×{pixel,normalized} to abs-xyxy.

    Returns clamped [x1,y1,x2,y2] ints, or None if degenerate/zero-area/malformed.
    """
    if raw is None:
        return None
    try:
        a, b, c, d = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError, IndexError):
        return None
    if any(math.isnan(v) or math.isinf(v) for v in (a, b, c, d)):
        return None

    normalized = fmt in ("xyxy_norm", "xywh_norm") or (
        fmt == "auto" and max(abs(a), abs(b), abs(c), abs(d)) <= 1.0 and W > 1 and H > 1)
    if normalized:
        a, c = a * W, c * W
        b, d = b * H, d * H

    if fmt in ("xywh", "xywh_norm"):
        is_xywh = True
    elif fmt in ("xyxy", "xyxy_norm"):
        is_xywh = False
    else:  # auto: xyxy unless it would invert (c<a or d<b), which only xywh explains
        is_xywh = (c < a) or (d < b)

    if is_xywh:
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        x1, y1, x2, y2 = a, b, c, d
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), W))
    y1 = max(0, min(int(round(y1)), H))
    x2 = max(0, min(int(round(x2)), W))
    y2 = max(0, min(int(round(y2)), H))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Display-label resolution (caption_mode / length tiers / template)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\{(\w+)(?::[^}]*)?\}")


def expand_label_template(template: str, ctx: dict) -> str:
    """Expand {token} / {token:fmt} against ctx; unknown tokens left blank, never raises."""
    if not template:
        return ""

    def repl(m):
        key = m.group(1)
        val = ctx.get(key, "")
        try:
            return ("{" + m.group(0)[1:]).format(**{key: val}) if ":" in m.group(0) else str(val)
        except (ValueError, KeyError, IndexError):
            return str(val)

    try:
        return _TOKEN_RE.sub(repl, template)
    except Exception:
        return template


def resolve_display_label(track: TrackAnnotation, caption_mode: str = "auto",
                          max_chars: Optional[int] = None) -> str:
    """Pick the label text for a track honoring caption_mode, with graceful fallback
    down the length tiers. Returns uppercase-ready raw text (caller may .upper())."""
    mode = (caption_mode or "auto").lower()
    if track.display_label and mode in ("auto", "mixed"):
        base = track.display_label
    elif mode == "short":
        base = track.short_label or track.medium_label or track.detailed_label or track.category
    elif mode == "medium":
        base = track.medium_label or track.short_label or track.detailed_label or track.category
    elif mode == "detailed":
        base = track.detailed_label or track.medium_label or track.short_label or track.category
    elif mode == "action":
        base = track.action_label or track.detailed_label or track.short_label or track.category
    elif mode == "technical":
        base = track.short_label or track.category
    else:  # auto/mixed without display_label
        base = (track.display_label or track.detailed_label or track.medium_label
                or track.short_label or track.category)
    base = base or track.category or "OBJECT"
    if max_chars and len(base) > max_chars > 1:
        base = base[: max_chars - 1].rstrip() + "…"
    return base


# ---------------------------------------------------------------------------
# Parsing / validation / normalization
# ---------------------------------------------------------------------------

def _as_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_child(d: dict, W: int, H: int, parent_id: Optional[str], report: list) -> Optional[ChildAnnotation]:
    if not isinstance(d, dict):
        report.append("child annotation is not an object; skipped")
        return None
    bbox_by_frame = {}
    raw_bbf = d.get("bbox_by_frame") or {}
    if isinstance(raw_bbf, dict):
        for k, v in raw_bbf.items():
            try:
                fi = int(k)
            except (TypeError, ValueError):
                continue
            bb = coerce_bbox(v, W, H, d.get("bbox_format", "auto"))
            if bb:
                bbox_by_frame[fi] = bb
    # single 'bbox' shorthand -> frame 0
    if not bbox_by_frame and d.get("bbox") is not None:
        bb = coerce_bbox(d.get("bbox"), W, H, d.get("bbox_format", "auto"))
        if bb:
            bbox_by_frame[int(d.get("frame_index", 0))] = bb
    return ChildAnnotation(
        annotation_id=str(d.get("annotation_id") or f"{parent_id or 'track'}:child"),
        category=str(d.get("category", "object")),
        short_label=filter_sensitive(d.get("short_label", "")),
        detailed_label=filter_sensitive(d.get("detailed_label", "")),
        confidence=_as_float(d.get("confidence")),
        region_source=str(d.get("region_source", "secondary_detector")),
        region_mode=str(d.get("region_mode", "bbox")),
        bbox_by_frame=bbox_by_frame,
        parent_track_id=d.get("parent_track_id", parent_id),
        priority=int(d.get("priority", 40)),
    )


def _coerce_track(d: dict, W: int, H: int, report: list) -> Optional[TrackAnnotation]:
    if not isinstance(d, dict):
        report.append("track annotation is not an object; skipped")
        return None
    tid = d.get("track_id")
    if not tid:
        cat = d.get("category", "object")
        idx = d.get("source_track_index", 0)
        tid = namespace_id(cat, idx)
        report.append(f"track missing track_id; generated '{tid}'")
    so_raw = d.get("style_overrides") or {}
    so = StyleOverrides(
        priority=int(so_raw.get("priority", 50)),
        preferred_color=so_raw.get("preferred_color"),
        preferred_asset=so_raw.get("preferred_asset"),
        preferred_label_mode=so_raw.get("preferred_label_mode"),
    )
    attrs_raw = d.get("attributes") or {}
    attrs = {k: _filter_attr_list(v) if isinstance(v, list) else filter_sensitive(str(v))
             for k, v in attrs_raw.items()} if isinstance(attrs_raw, dict) else {}
    children = []
    for c in (d.get("children") or []):
        cc = _coerce_child(c, W, H, str(tid), report)
        if cc:
            children.append(cc)
    return TrackAnnotation(
        track_id=str(tid),
        source=str(d.get("source", "sam3_primary")),
        source_track_index=d.get("source_track_index"),
        category=str(d.get("category", "object")),
        short_label=filter_sensitive(d.get("short_label", "")),
        medium_label=filter_sensitive(d.get("medium_label", "")),
        detailed_label=filter_sensitive(d.get("detailed_label", "")),
        action_label=filter_sensitive(d.get("action_label", "")),
        context_label=filter_sensitive(d.get("context_label", "")),
        display_label=filter_sensitive(d.get("display_label", "")),
        confidence=_as_float(d.get("confidence")),
        attributes=attrs,
        style_overrides=so,
        children=children,
    )


def namespace_id(group_or_category: str, index) -> str:
    """Build a globally-unique namespaced id, e.g. ('person', 0) -> 'person:0'."""
    g = re.sub(r"[^a-zA-Z0-9_]+", "_", str(group_or_category or "track")).strip("_").lower() or "track"
    return f"{g}:{index}"


def normalize_annotation_set(obj: dict, W: int = 0, H: int = 0, frame_count: int = 0) -> tuple:
    """Coerce a raw dict into a validated AnnotationSet. Returns (AnnotationSet, report:list[str])."""
    report: list = []
    if not isinstance(obj, dict):
        return AnnotationSet(), ["root is not a JSON object"]
    vid_raw = obj.get("video") or {}
    vid = VideoMeta(
        width=int(vid_raw.get("width", W) or W),
        height=int(vid_raw.get("height", H) or H),
        fps=_as_float(vid_raw.get("fps"), 0.0) or 0.0,
        frame_count=int(vid_raw.get("frame_count", frame_count) or frame_count),
    )
    use_w = vid.width or W or 1920
    use_h = vid.height or H or 1080

    tracks = []
    seen_ids = set()
    for t in (obj.get("tracks") or []):
        tr = _coerce_track(t, use_w, use_h, report)
        if tr is None:
            continue
        if tr.track_id in seen_ids:
            report.append(f"duplicate track_id '{tr.track_id}'; keeping first")
            continue
        seen_ids.add(tr.track_id)
        tracks.append(tr)

    frame_annos: dict = {}
    fa_raw = obj.get("frame_annotations") or {}
    if isinstance(fa_raw, dict):
        for k, lst in fa_raw.items():
            try:
                fi = int(k)
            except (TypeError, ValueError):
                report.append(f"non-integer frame key '{k}'; skipped")
                continue
            items = []
            for a in (lst or []):
                if not isinstance(a, dict):
                    continue
                bb = coerce_bbox(a.get("bbox"), use_w, use_h, a.get("bbox_format", "auto"))
                if bb is None:
                    report.append(f"frame {fi}: annotation '{a.get('annotation_id')}' bbox invalid; skipped")
                    continue
                items.append(FrameAnnotation(
                    annotation_id=str(a.get("annotation_id") or f"object:{fi}:{len(items)}"),
                    category=str(a.get("category", "object")),
                    short_label=filter_sensitive(a.get("short_label", "")),
                    detailed_label=filter_sensitive(a.get("detailed_label", "")),
                    bbox=bb, confidence=_as_float(a.get("confidence")),
                    priority=int(a.get("priority", 60)),
                    parent_track_id=a.get("parent_track_id"), frame_index=fi))
            if items:
                frame_annos[fi] = items

    aset = AnnotationSet(version=int(obj.get("version", SCHEMA_VERSION)), video=vid,
                         tracks=tracks, frame_annotations=frame_annos)
    return aset, report


def parse_annotations_json(s, W: int = 0, H: int = 0, frame_count: int = 0,
                           strict: bool = False) -> tuple:
    """Parse + validate + normalize an annotations JSON string (or dict/AnnotationSet).
    Returns (AnnotationSet, report:list[str]). On invalid JSON: empty set + report
    (or raises ValueError if strict)."""
    if isinstance(s, AnnotationSet):
        return s, []
    if isinstance(s, dict):
        return normalize_annotation_set(s, W, H, frame_count)
    if not s or not str(s).strip():
        return AnnotationSet(video=VideoMeta(width=W, height=H, frame_count=frame_count)), []
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        msg = f"invalid annotations JSON at line {exc.lineno} col {exc.colno}: {exc.msg}"
        if strict:
            raise ValueError(msg) from exc
        return AnnotationSet(video=VideoMeta(width=W, height=H, frame_count=frame_count)), [
            msg, 'example: {"version":1,"tracks":[{"track_id":"person:0","short_label":"PERSON"}]}']
    return normalize_annotation_set(obj, W, H, frame_count)


# ---------------------------------------------------------------------------
# Build from native tracking + merge multiple sources
# ---------------------------------------------------------------------------

def annotation_set_from_detections(per_frame_dets, group_label: str = "PERSON",
                                   category: str = "person", source: str = "sam3_primary",
                                   W: int = 0, H: int = 0, frame_count: int = 0) -> AnnotationSet:
    """Build a generic AnnotationSet from per-frame Detections (tracking layer output).
    Each distinct source_track_id becomes a TrackAnnotation with a safe generic label."""
    tracks: dict = {}
    for f, dets in enumerate(per_frame_dets or []):
        for det in dets:
            stid = det.source_track_id if det.source_track_id is not None else det.source_index
            tid = namespace_id(category, stid)
            if tid not in tracks:
                tracks[tid] = TrackAnnotation(
                    track_id=tid, source=source, source_track_index=(
                        int(stid) if isinstance(stid, int) else None),
                    category=category, short_label=group_label.upper(),
                    display_label=group_label.upper(),
                    confidence=det.score,
                    style_overrides=StyleOverrides(priority=100 if category == "person" else 60))
    return AnnotationSet(video=VideoMeta(width=W, height=H, frame_count=frame_count or len(per_frame_dets or [])),
                         tracks=list(tracks.values()))


def merge_annotation_sources(base: Optional[AnnotationSet], captions_json: str = "",
                             manual_json: str = "", secondary_json: str = "",
                             default_category: str = "person",
                             label_template: str = "", W: int = 0, H: int = 0) -> tuple:
    """Merge a base set (from tracking) with caption JSON (by-track labels), manual JSON
    (full annotation set) and secondary detections JSON (frame_annotations). Manual JSON
    wins on conflicts; caption JSON fills labels; secondary adds frame_annotations.
    Returns (AnnotationSet, report)."""
    report: list = []
    aset = base if base is not None else AnnotationSet(video=VideoMeta(width=W, height=H))
    by_id = aset.tracks_by_id()

    # 1) caption JSON: {"person:0":"SHIRTLESS MAN ..."} OR full track list
    if captions_json and captions_json.strip():
        try:
            cap = json.loads(captions_json)
        except json.JSONDecodeError as exc:
            report.append(f"captions_json invalid: {exc.msg}")
            cap = None
        if isinstance(cap, dict) and "tracks" in cap:
            cset, r = normalize_annotation_set(cap, W, H)
            report += r
            for t in cset.tracks:
                _merge_track_labels(by_id, aset, t, report)
        elif isinstance(cap, dict):
            for tid, label in cap.items():
                t = by_id.get(tid)
                if t is None:
                    t = TrackAnnotation(track_id=str(tid), category=default_category)
                    aset.tracks.append(t); by_id[t.track_id] = t
                if isinstance(label, dict):
                    _apply_label_fields(t, label)
                else:
                    t.detailed_label = filter_sensitive(str(label))
                    if not t.display_label:
                        t.display_label = t.detailed_label

    # 2) manual JSON: a full annotation set, authoritative
    if manual_json and manual_json.strip():
        mset, r = parse_annotations_json(manual_json, W, H)
        report += [f"manual: {x}" for x in r]
        for t in mset.tracks:
            _merge_track_labels(by_id, aset, t, report, authoritative=True)
        for fi, lst in mset.frame_annotations.items():
            aset.frame_annotations.setdefault(fi, []).extend(lst)

    # 3) secondary detections JSON: frame_annotations / universal detection list
    if secondary_json and secondary_json.strip():
        try:
            sec = json.loads(secondary_json)
        except json.JSONDecodeError as exc:
            report.append(f"secondary_detections_json invalid: {exc.msg}")
            sec = None
        if isinstance(sec, list):
            _, sset = adapt_universal_detections(sec, W or aset.video.width, H or aset.video.height)
            for fi, lst in sset.frame_annotations.items():
                aset.frame_annotations.setdefault(fi, []).extend(lst)
        elif isinstance(sec, dict):
            sset, r = normalize_annotation_set(sec, W, H)
            report += r
            for fi, lst in sset.frame_annotations.items():
                aset.frame_annotations.setdefault(fi, []).extend(lst)

    # 4) apply a label_template to any track still lacking a display_label
    if label_template:
        for t in aset.tracks:
            if not t.display_label:
                t.display_label = expand_label_template(label_template, {
                    "id": t.track_id, "short_label": t.short_label, "medium_label": t.medium_label,
                    "detailed_label": t.detailed_label, "action_label": t.action_label,
                    "category": t.category,
                    "score_percent": int((t.confidence or 0) * 100)})
    return aset, report


def _apply_label_fields(t: TrackAnnotation, d: dict):
    for fld in ("short_label", "medium_label", "detailed_label", "action_label",
                "context_label", "display_label"):
        if d.get(fld):
            setattr(t, fld, filter_sensitive(d[fld]))
    if d.get("attributes") and isinstance(d["attributes"], dict):
        t.attributes.update({k: _filter_attr_list(v) if isinstance(v, list) else filter_sensitive(str(v))
                             for k, v in d["attributes"].items()})
    if not t.display_label:
        t.display_label = t.detailed_label or t.medium_label or t.short_label


def _merge_track_labels(by_id, aset, t: TrackAnnotation, report, authoritative=False):
    existing = by_id.get(t.track_id)
    if existing is None:
        aset.tracks.append(t); by_id[t.track_id] = t
        return
    for fld in ("short_label", "medium_label", "detailed_label", "action_label",
                "context_label", "display_label"):
        new = getattr(t, fld)
        if new and (authoritative or not getattr(existing, fld)):
            setattr(existing, fld, new)
    if t.attributes:
        existing.attributes.update(t.attributes)
    if t.children:
        existing.children.extend(t.children)
    if not existing.display_label:
        existing.display_label = existing.detailed_label or existing.medium_label or existing.short_label


# ---------------------------------------------------------------------------
# Universal external detection JSON adapter (does NOT replace SAM3 tracking)
# ---------------------------------------------------------------------------

def adapt_universal_detections(entries, W: int, H: int) -> tuple:
    """Accept a list of {frame_index, track_id?, bbox, bbox_format?, label?, score?,
    category?, source?, parent_track_id?} dicts. Returns (per_frame_detections, AnnotationSet
    of frame_annotations). Malformed entries are reported, not fatal."""
    from .types import Detection  # local import keeps this module ComfyUI/torch-free at load

    report: list = []
    per_frame: dict = {}
    frame_annos: dict = {}
    auto_idx = 0
    for e in (entries or []):
        if not isinstance(e, dict):
            continue
        fi = int(e.get("frame_index", 0))
        bb = coerce_bbox(e.get("bbox"), W, H, e.get("bbox_format", "auto"))
        if bb is None:
            report.append(f"frame {fi}: zero-area/invalid bbox; skipped")
            continue
        tid = e.get("track_id")
        if not tid:
            tid = namespace_id(e.get("category", "object"), auto_idx); auto_idx += 1
        x1, y1, x2, y2 = bb
        per_frame.setdefault(fi, []).append(Detection(
            frame_index=fi, source_index=len(per_frame.get(fi, [])), x1=x1, y1=y1, x2=x2, y2=y2,
            score=_as_float(e.get("score")), label=filter_sensitive(e.get("label", "")) or None,
            source_track_id=str(tid)))
        frame_annos.setdefault(fi, []).append(FrameAnnotation(
            annotation_id=str(tid), category=str(e.get("category", "object")),
            short_label=filter_sensitive(e.get("short_label", "")),
            detailed_label=filter_sensitive(e.get("label", "")),
            bbox=bb, confidence=_as_float(e.get("score")), parent_track_id=e.get("parent_track_id"),
            frame_index=fi))
    max_f = (max(per_frame) + 1) if per_frame else 0
    dense = [per_frame.get(i, []) for i in range(max_f)]
    return dense, AnnotationSet(video=VideoMeta(width=W, height=H, frame_count=max_f),
                                frame_annotations=frame_annos)


# ---------------------------------------------------------------------------
# Caption consensus across multiple representative crops of one track
# ---------------------------------------------------------------------------

def caption_consensus(samples: list) -> dict:
    """Merge several VLM JSON descriptions of the SAME track into one stable result.
    Prefers attributes confirmed in >1 sample, drops contradictions, picks one action.
    Returns a dict with short/medium/detailed/action/context labels + merged attributes."""
    samples = [s for s in (samples or []) if isinstance(s, dict)]
    if not samples:
        return {}
    if len(samples) == 1:
        return _sanitize_vlm(samples[0])

    def tally(field_name):
        counts = {}
        for s in samples:
            for v in (s.get("attributes", {}).get(field_name, []) or []):
                key = filter_sensitive(str(v)).lower().strip()
                if key:
                    counts[key] = counts.get(key, 0) + 1
        # prefer confirmed-in-multiple, then by frequency, stable order
        confirmed = sorted([k for k, c in counts.items() if c >= 2], key=lambda k: -counts[k])
        singles = sorted([k for k, c in counts.items() if c == 1])
        return confirmed + singles

    merged_attrs = {f: tally(f) for f in ("clothing", "visible_details", "action", "objects", "scene")}
    # pick the most common non-empty label per tier
    def pick(field_name):
        vals = [filter_sensitive(s.get(field_name, "")) for s in samples if s.get(field_name)]
        if not vals:
            return ""
        freq = {}
        for v in vals:
            freq[v] = freq.get(v, 0) + 1
        return max(freq, key=lambda k: (freq[k], -len(k)))

    out = {
        "short_label": pick("short_label"),
        "medium_label": pick("medium_label"),
        "detailed_label": pick("detailed_label"),
        "action_label": pick("action_label") or (merged_attrs["action"][0].upper() if merged_attrs["action"] else ""),
        "context_label": pick("context_label"),
        "category": pick("category") or samples[0].get("category", "object"),
        "attributes": merged_attrs,
        "confidence": sum(_as_float(s.get("confidence"), 0.0) or 0.0 for s in samples) / len(samples),
    }
    out["display_label"] = out["detailed_label"] or out["medium_label"] or out["short_label"]
    return _sanitize_vlm(out)


def _sanitize_vlm(d: dict) -> dict:
    out = dict(d)
    for fld in ("short_label", "medium_label", "detailed_label", "action_label",
                "context_label", "display_label"):
        if out.get(fld):
            out[fld] = filter_sensitive(out[fld])
    attrs = out.get("attributes")
    if isinstance(attrs, dict):
        out["attributes"] = {k: _filter_attr_list(v) if isinstance(v, list) else filter_sensitive(str(v))
                             for k, v in attrs.items()}
    if not out.get("display_label"):
        out["display_label"] = out.get("detailed_label") or out.get("medium_label") or out.get("short_label", "")
    return out


__all__ = [
    "SCHEMA_VERSION", "ANNOTATIONS_TYPE", "CAPTION_MODES", "LABEL_TIERS",
    "VideoMeta", "StyleOverrides", "ChildAnnotation", "TrackAnnotation", "FrameAnnotation",
    "AnnotationSet", "filter_sensitive", "coerce_bbox", "namespace_id",
    "expand_label_template", "resolve_display_label", "parse_annotations_json",
    "normalize_annotation_set", "annotation_set_from_detections", "merge_annotation_sources",
    "adapt_universal_detections", "caption_consensus",
]
