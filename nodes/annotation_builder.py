"""Yonder Annotation Builder — merge tracking + labels into one YONDER_ANNOTATIONS.

Combines: a base annotation set derived from SAM3 track_data (generic group labels +
stable namespaced ids), VLM responses (per-crop list reduced to one consensus label
per track, or a by-track dict), manual JSON (authoritative), and secondary detection
JSON (frame_annotations). No network, no model — pure merge/validate.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("YonderHUD.annotation_builder")

DISPLAY_NAME = "Yonder Annotation Builder"
NODE_ID = "YonderAnnotationBuilder"
CATEGORY = "Yonder/HUD/Annotation"
ANNOTATIONS_TYPE = "YONDER_ANNOTATIONS"


def _consensus_from_vlm_list(captions_json, crop_manifest_json, report):
    """If captions_json is a LIST of per-crop VLM responses, reduce to one consensus
    label per track (resolving crop_index -> track_id via the manifest). Returns a
    by-track dict JSON string suitable for merging, or the original string unchanged."""
    from yonder_hud.annotations import caption_consensus

    if not captions_json or not captions_json.strip().startswith("["):
        return captions_json
    try:
        vlm_list = json.loads(captions_json)
    except json.JSONDecodeError as exc:
        report.append(f"captions VLM list invalid: {exc.msg}")
        return ""
    cmap = {}
    if crop_manifest_json and crop_manifest_json.strip():
        try:
            manifest = json.loads(crop_manifest_json)
            for entry in (manifest.get("crops", []) if isinstance(manifest, dict) else []):
                if "crop_index" in entry:
                    cmap[entry["crop_index"]] = entry.get("track_id")
        except json.JSONDecodeError as exc:
            report.append(f"crop_manifest invalid: {exc.msg}")
    by_track = {}
    for r in vlm_list:
        if not isinstance(r, dict):
            continue
        tid = r.get("track_id") or cmap.get(r.get("crop_index"))
        if tid:
            by_track.setdefault(str(tid), []).append(r)
    consensus = {tid: caption_consensus(samples) for tid, samples in by_track.items()}
    report.append(f"reduced {len(vlm_list)} VLM responses -> {len(consensus)} track labels")
    return json.dumps(consensus)


def run_annotation_builder(track_data=None, crop_manifest_json="", captions_json="",
                           manual_annotations_json="", secondary_detections_json="",
                           default_category="person", label_template="", primary_group_label="PERSON"):
    """Return (AnnotationSet, annotations_json_str, validation_report_str)."""
    from yonder_hud import track_adapter
    from yonder_hud.annotations import annotation_set_from_detections, merge_annotation_sources

    report = []
    W = H = fc = 0
    base = None
    if track_data is not None and track_adapter.is_track_data(track_data):
        dets, n_obj = track_adapter.adapt_track_data(track_data)
        osz = track_data.get("orig_size") or (0, 0)
        try:
            H, W = int(osz[0]), int(osz[1])
        except (TypeError, ValueError, IndexError):
            H = W = 0
        fc = len(dets)
        base = annotation_set_from_detections(dets, group_label=primary_group_label,
                                              category=default_category, W=W, H=H, frame_count=fc)
        report.append(f"base from track_data: {n_obj} tracks, {fc} frames")

    captions_for_merge = _consensus_from_vlm_list(captions_json, crop_manifest_json, report)

    merged, merge_report = merge_annotation_sources(
        base, captions_json=captions_for_merge, manual_json=manual_annotations_json,
        secondary_json=secondary_detections_json, default_category=default_category,
        label_template=label_template, W=W, H=H)
    report += merge_report

    labelled = sum(1 for t in merged.tracks if t.display_label)
    summary = [f"tracks: {len(merged.tracks)} ({labelled} labelled)",
               f"frame annotations: {sum(len(v) for v in merged.frame_annotations.values())}",
               f"notes: {len(report)}"] + [f"  - {r}" for r in report[:30]]
    return merged, merged.to_json(indent=2), "\n".join(summary)


try:
    from comfy_api.latest import io

    YonderAnnotations = io.Custom(ANNOTATIONS_TYPE)
    SAM3TrackData = io.Custom("SAM3_TRACK_DATA")

    class YonderAnnotationBuilder(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Merge SAM3 track ids, VLM responses, manual labels and secondary detections "
                    "into one stable YONDER_ANNOTATIONS object. VLM crop responses are reduced to "
                    "one consensus label per track. No network / no model here."
                ),
                search_aliases=["annotation", "merge", "builder", "vlm", "label"],
                inputs=[
                    SAM3TrackData.Input("track_data", optional=True,
                                        tooltip="Base identities from SAM3 Video Track."),
                    io.String.Input("crop_manifest_json", default="", multiline=True, optional=True,
                                    tooltip="From Track Crop Extractor; maps crop_index -> track_id."),
                    io.String.Input("captions_json", default="", multiline=True, optional=True,
                                    tooltip='VLM output: a list of per-crop responses, OR a by-track dict {"person:0":"..."}.'),
                    io.String.Input("manual_annotations_json", default="", multiline=True, optional=True,
                                    tooltip="Authoritative manual YONDER_ANNOTATIONS JSON (wins on conflicts)."),
                    io.String.Input("secondary_detections_json", default="", multiline=True, optional=True,
                                    tooltip="Secondary detector boxes (frame_annotations or universal list)."),
                    io.String.Input("default_category", default="person", optional=True),
                    io.String.Input("primary_group_label", default="PERSON", optional=True),
                    io.String.Input("label_template", default="", optional=True,
                                    tooltip="Optional template for tracks lacking a display_label, e.g. '{id}. {short_label}'."),
                ],
                outputs=[
                    YonderAnnotations.Output("annotations", display_name="annotations"),
                    io.String.Output("annotations_json", display_name="annotations_json"),
                    io.String.Output("validation_report", display_name="validation_report"),
                ],
            )

        @classmethod
        def execute(cls, track_data=None, crop_manifest_json="", captions_json="",
                    manual_annotations_json="", secondary_detections_json="",
                    default_category="person", primary_group_label="PERSON",
                    label_template="") -> io.NodeOutput:
            aset, aj, report = run_annotation_builder(
                track_data, crop_manifest_json, captions_json, manual_annotations_json,
                secondary_detections_json, default_category, label_template, primary_group_label)
            return io.NodeOutput(aset, aj, report)

    NODE = YonderAnnotationBuilder
except ImportError:  # pragma: no cover
    NODE = None
