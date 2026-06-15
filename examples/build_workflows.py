"""Generate + validate importable ComfyUI workflow JSON (litegraph UI format).

Dev tool (not shipped to the Registry — see .comfyignore). Introspects the V3
schemas of the Yonder nodes for their canonical socket order, hardcodes the native
ComfyUI node socket specs (verified from comfy_extras source), builds the graphs
with correct slot indices, VALIDATES that every link connects type-matching sockets
and that every node type is registered, then writes the .json files next to this
script.

Run:
  PYTHONPATH=<comfyui-ref> /path/to/.venv/bin/python examples/build_workflows.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Native ComfyUI node specs (verified from comfy_extras/nodes_video.py, nodes.py,
# comfy_extras/nodes_sam3.py). socket inputs in display order; widgets (name,default);
# outputs in order.
NATIVE = {
    "CheckpointLoaderSimple": {"in": [], "widgets": [("ckpt_name", "sam3.safetensors")],
                               "out": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")]},
    "LoadVideo": {"in": [], "widgets": [("file", "input.mp4")], "out": [("VIDEO", "VIDEO")]},
    "GetVideoComponents": {"in": [("video", "VIDEO")], "widgets": [],
                           "out": [("IMAGE", "IMAGE"), ("AUDIO", "AUDIO"), ("FLOAT", "FLOAT"), ("INT", "INT")]},
    "CLIPTextEncode": {"in": [("clip", "CLIP")], "widgets": [("text", "person")],
                       "out": [("CONDITIONING", "CONDITIONING")]},
    "SAM3_VideoTrack": {"in": [("images", "IMAGE"), ("model", "MODEL"), ("initial_mask", "MASK"),
                               ("conditioning", "CONDITIONING")],
                        "widgets": [("detection_threshold", 0.5), ("max_objects", 8), ("detect_interval", 1)],
                        "out": [("track_data", "SAM3_TRACK_DATA")]},
    "SAM3_Detect": {"in": [("model", "MODEL"), ("image", "IMAGE"), ("conditioning", "CONDITIONING"),
                           ("bboxes", "BOUNDING_BOX"), ("positive_coords", "STRING"), ("negative_coords", "STRING")],
                    "widgets": [("threshold", 0.5), ("refine_iterations", 2), ("individual_masks", True)],
                    "out": [("masks", "MASK"), ("bboxes", "BOUNDING_BOX")]},
    "CreateVideo": {"in": [("images", "IMAGE"), ("audio", "AUDIO")], "widgets": [("fps", 24.0), ("bit_depth", 8)],
                    "out": [("VIDEO", "VIDEO")]},
    "SaveVideo": {"in": [("video", "VIDEO")], "widgets": [("filename_prefix", "video/YonderHUD"),
                                                          ("format", "auto"), ("codec", "auto")], "out": []},
    "LoadImage": {"in": [], "widgets": [("image", "frame.png")], "out": [("IMAGE", "IMAGE"), ("MASK", "MASK")]},
}


def _load_my_specs():
    from comfy_api.latest import io  # noqa: F401
    from nodes.hud_renderer import YonderHUDRenderer
    from nodes.inspect_track_data import YonderInspectSAM3TrackData
    from nodes.inspect_bbox import YonderInspectBoundingBoxes
    from nodes.mask_to_bbox import YonderMaskToNativeBBoxes
    from nodes.annotation_json import YonderAnnotationJSON
    from nodes.annotation_builder import YonderAnnotationBuilder
    from nodes.track_crop_extractor import YonderTrackCropExtractor
    from nodes.detailed_label_prompt import YonderDetailedLabelPrompt
    classes = [YonderHUDRenderer, YonderInspectSAM3TrackData, YonderInspectBoundingBoxes,
               YonderMaskToNativeBBoxes, YonderAnnotationJSON, YonderAnnotationBuilder,
               YonderTrackCropExtractor, YonderDetailedLabelPrompt]
    specs = {}
    for cls in classes:
        s = cls.define_schema()
        sockets, widgets = [], []
        for inp in s.inputs:
            iot = inp.io_type
            is_socket = (iot not in WIDGET_TYPES) or bool(getattr(inp, "force_input", False))
            if is_socket:
                sockets.append((inp.id, iot))
            else:
                widgets.append((inp.id, getattr(inp, "default", None)))
        outs = [(o.id, o.io_type) for o in s.outputs]
        specs[s.node_id] = {"in": sockets, "widgets": widgets, "out": outs}
    return specs


class Graph:
    def __init__(self, specs):
        self.specs = specs
        self.nodes = []
        self.links = []
        self._by_id = {}
        self._nid = 0
        self._lid = 0

    def add(self, ntype, *, widget_values=None, pos=(0, 0), title=None):
        if ntype not in self.specs:
            raise KeyError(f"unknown node type {ntype!r} (not registered)")
        spec = self.specs[ntype]
        self._nid += 1
        nid = self._nid
        wv = []
        for i, (_wn, wd) in enumerate(spec["widgets"]):
            wv.append(widget_values[i] if widget_values and i < len(widget_values) else wd)
        node = {
            "id": nid, "type": ntype, "pos": [pos[0], pos[1]], "size": [300, 120],
            "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for (n, t) in spec["in"]],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(spec["out"])],
            "properties": {"Node name for S&R": ntype},
            "widgets_values": wv,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        self._by_id[nid] = node
        return nid

    def _out_slot(self, nid, name):
        for i, o in enumerate(self._by_id[nid]["outputs"]):
            if o["name"] == name or o["type"] == name:
                return i, o["type"]
        raise KeyError(f"output {name!r} not on node {self._by_id[nid]['type']}")

    def _in_slot(self, nid, name):
        for i, inp in enumerate(self._by_id[nid]["inputs"]):
            if inp["name"] == name:
                return i, inp["type"]
        raise KeyError(f"input {name!r} not a socket on node {self._by_id[nid]['type']}")

    def connect(self, src_id, out_name, dst_id, in_name):
        so, st = self._out_slot(src_id, out_name)
        di, dt = self._in_slot(dst_id, in_name)
        if st != dt:
            raise TypeError(f"socket type mismatch: {self._by_id[src_id]['type']}.{out_name}({st}) "
                            f"-> {self._by_id[dst_id]['type']}.{in_name}({dt})")
        self._lid += 1
        lid = self._lid
        self.links.append([lid, src_id, so, dst_id, di, st])
        self._by_id[src_id]["outputs"][so]["links"].append(lid)
        self._by_id[dst_id]["inputs"][di]["link"] = lid
        return lid

    def connect_widget(self, src_id, out_name, dst_id, widget_name, wtype):
        """Wire an output into a destination WIDGET (converted to an input)."""
        so, st = self._out_slot(src_id, out_name)
        if st != wtype:
            raise TypeError(f"widget-input type mismatch: {out_name}({st}) -> {widget_name}({wtype})")
        self._lid += 1
        lid = self._lid
        dst = self._by_id[dst_id]
        dst["inputs"].append({"name": widget_name, "type": wtype, "link": lid,
                              "widget": {"name": widget_name}})
        self.links.append([lid, src_id, so, dst_id, len(dst["inputs"]) - 1, st])
        self._by_id[src_id]["outputs"][so]["links"].append(lid)
        return lid

    def to_dict(self):
        return {"last_node_id": self._nid, "last_link_id": self._lid, "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


def _renderer_widget_values(specs):
    """Default widget values for the renderer (positional, schema order)."""
    return [w[1] for w in specs["YonderHUDRenderer"]["widgets"]]


def build_video(specs):
    g = Graph(specs)
    ckpt = g.add("CheckpointLoaderSimple", pos=(0, 0))
    lv = g.add("LoadVideo", pos=(0, 200))
    gvc = g.add("GetVideoComponents", pos=(340, 200))
    clip = g.add("CLIPTextEncode", pos=(340, 0), widget_values=["person"])
    track = g.add("SAM3_VideoTrack", pos=(700, 100), widget_values=[0.5, 8, 1])
    rend = g.add("YonderHUDRenderer", pos=(1060, 100), widget_values=_renderer_widget_values(specs))
    cv = g.add("CreateVideo", pos=(1420, 100), widget_values=[24.0, 8])
    sv = g.add("SaveVideo", pos=(1760, 100))
    g.connect(ckpt, "MODEL", track, "model")
    g.connect(ckpt, "CLIP", clip, "clip")
    g.connect(lv, "VIDEO", gvc, "video")
    g.connect(gvc, "IMAGE", track, "images")
    g.connect(clip, "CONDITIONING", track, "conditioning")
    g.connect(gvc, "IMAGE", rend, "frames")
    g.connect(track, "track_data", rend, "track_data")
    g.connect(rend, "images", cv, "images")
    g.connect(gvc, "AUDIO", cv, "audio")
    g.connect_widget(gvc, "FLOAT", cv, "fps", "FLOAT")   # preserve original fps
    g.connect(cv, "VIDEO", sv, "video")
    return g


def build_image_batch(specs):
    g = Graph(specs)
    ckpt = g.add("CheckpointLoaderSimple", pos=(0, 0))
    lv = g.add("LoadVideo", pos=(0, 200))
    gvc = g.add("GetVideoComponents", pos=(340, 200))
    clip = g.add("CLIPTextEncode", pos=(340, 0), widget_values=["person"])
    det = g.add("SAM3_Detect", pos=(700, 100), widget_values=[0.5, 2, True])
    rend = g.add("YonderHUDRenderer", pos=(1060, 100), widget_values=_renderer_widget_values(specs))
    cv = g.add("CreateVideo", pos=(1420, 100), widget_values=[24.0, 8])
    sv = g.add("SaveVideo", pos=(1760, 100))
    g.connect(ckpt, "MODEL", det, "model")
    g.connect(ckpt, "CLIP", clip, "clip")
    g.connect(lv, "VIDEO", gvc, "video")
    g.connect(gvc, "IMAGE", det, "image")
    g.connect(clip, "CONDITIONING", det, "conditioning")
    g.connect(gvc, "IMAGE", rend, "frames")
    g.connect(det, "bboxes", rend, "bboxes")
    g.connect(det, "masks", rend, "masks")
    g.connect(rend, "images", cv, "images")
    g.connect(gvc, "AUDIO", cv, "audio")
    g.connect_widget(gvc, "FLOAT", cv, "fps", "FLOAT")
    g.connect(cv, "VIDEO", sv, "video")
    return g


def build_manual_annotation(specs):
    g = build_video(specs)
    # add an Annotation JSON node feeding the renderer's `annotations` input.
    aj = g.add("YonderAnnotationJSON", pos=(700, 360))
    rend = next(n["id"] for n in g.nodes if n["type"] == "YonderHUDRenderer")
    g.connect(aj, "annotations", rend, "annotations")
    return g


def build_vlm_annotation(specs):
    g = Graph(specs)
    ckpt = g.add("CheckpointLoaderSimple", pos=(0, 0))
    lv = g.add("LoadVideo", pos=(0, 220))
    gvc = g.add("GetVideoComponents", pos=(340, 220))
    clip = g.add("CLIPTextEncode", pos=(340, 0), widget_values=["person"])
    track = g.add("SAM3_VideoTrack", pos=(700, 120), widget_values=[0.5, 8, 1])
    crop = g.add("YonderTrackCropExtractor", pos=(1060, 320), widget_values=[3, 0.15, 0.0, "best"])
    prompt = g.add("YonderDetailedLabelPrompt", pos=(1060, 520))
    builder = g.add("YonderAnnotationBuilder", pos=(1420, 320))
    rend = g.add("YonderHUDRenderer", pos=(1420, 80), widget_values=_renderer_widget_values(specs))
    cv = g.add("CreateVideo", pos=(1780, 80), widget_values=[24.0, 8])
    sv = g.add("SaveVideo", pos=(2120, 80))
    g.connect(ckpt, "MODEL", track, "model")
    g.connect(ckpt, "CLIP", clip, "clip")
    g.connect(lv, "VIDEO", gvc, "video")
    g.connect(gvc, "IMAGE", track, "images")
    g.connect(clip, "CONDITIONING", track, "conditioning")
    g.connect(gvc, "IMAGE", crop, "frames")
    g.connect(track, "track_data", crop, "track_data")
    g.connect(track, "track_data", builder, "track_data")
    g.connect_widget(crop, "crop_manifest_json", builder, "crop_manifest_json", "STRING")
    # NOTE: a VLM/partner node consumes crop + prompt and feeds builder.captions_json (user wires it).
    g.connect(builder, "annotations", rend, "annotations")
    g.connect(gvc, "IMAGE", rend, "frames")
    g.connect(track, "track_data", rend, "track_data")
    g.connect(rend, "images", cv, "images")
    g.connect(gvc, "AUDIO", cv, "audio")
    g.connect_widget(gvc, "FLOAT", cv, "fps", "FLOAT")
    g.connect(cv, "VIDEO", sv, "video")
    return g


def build_dense_multi(specs):
    g = build_video(specs)
    rend = next(n["id"] for n in g.nodes if n["type"] == "YonderHUDRenderer")
    # a second SAM3 group (e.g. faces) -> renderer.secondary_track_data_1
    ckpt = next(n["id"] for n in g.nodes if n["type"] == "CheckpointLoaderSimple")
    gvc = next(n["id"] for n in g.nodes if n["type"] == "GetVideoComponents")
    clip2 = g.add("CLIPTextEncode", pos=(340, 380), widget_values=["face"])
    track2 = g.add("SAM3_VideoTrack", pos=(700, 380), widget_values=[0.5, 8, 1])
    g.connect(ckpt, "CLIP", clip2, "clip")
    g.connect(ckpt, "MODEL", track2, "model")
    g.connect(gvc, "IMAGE", track2, "images")
    g.connect(clip2, "CONDITIONING", track2, "conditioning")
    g.connect(track2, "track_data", rend, "secondary_track_data_1")
    return g


def validate(graph_dict, specs):
    """Re-check: every node type registered; every link type-consistent."""
    issues = []
    type_by_id = {n["id"]: n["type"] for n in graph_dict["nodes"]}
    for n in graph_dict["nodes"]:
        if n["type"] not in specs:
            issues.append(f"unregistered node type: {n['type']}")
    for ln in graph_dict["links"]:
        lid, src, so, dst, di, t = ln
        st = type_by_id.get(src)
        if st and so < len(specs[st]["out"]):
            if specs[st]["out"][so][1] != t:
                issues.append(f"link {lid}: declared type {t} != source out type {specs[st]['out'][so][1]}")
    return issues


def main():
    specs = _load_my_specs()
    specs.update(NATIVE)
    builders = {
        "sam3_hud_video_workflow.json": build_video,
        "sam3_hud_image_batch_workflow.json": build_image_batch,
        "manual_annotation_workflow.json": build_manual_annotation,
        "vlm_annotation_workflow.json": build_vlm_annotation,
        "dense_multi_category_workflow.json": build_dense_multi,
    }
    all_ok = True
    for fn, builder in builders.items():
        g = builder(specs)
        d = g.to_dict()
        issues = validate(d, specs)
        status = "OK" if not issues else f"{len(issues)} ISSUES"
        print(f"{fn}: {len(d['nodes'])} nodes, {len(d['links'])} links -> {status}")
        for i in issues:
            print("   -", i)
            all_ok = False
        with open(os.path.join(_HERE, fn), "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
    print("ALL VALID" if all_ok else "VALIDATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
