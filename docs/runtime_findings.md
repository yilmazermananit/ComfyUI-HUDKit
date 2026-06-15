# Runtime Findings — ComfyUI-YonderHUD Milestone 1

> Bu doküman, renderer kodu yazılmadan **önce** gerçek ComfyUI kaynak kodundan
> doğrulanan veri yapılarını içerir. Hiçbir yapı tahmin edilmemiştir; her madde
> kaynak dosya + satır referanslıdır. Referans kaynak: bu makineye klonlanan
> `comfyanonymous/ComfyUI` master.

## 0. Ortam

| Öğe | Değer |
|-----|-------|
| Aktif ComfyUI çalıştırıcısı | **Comfy Desktop.app** (Electron, bootstrap-python 3.13) |
| Çekirdek `main.py` (bu sabahki log) | `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/main.py` (app güncellemesiyle silinmiş) |
| `basePath` (custom_nodes/models/input/output/user) | `/Users/differenterman/Documents/ComfyUI` |
| Python venv (çekirdeği çalıştıran) | `/Users/differenterman/Documents/ComfyUI/.venv/bin/python` → CPython 3.12.8 (uv) |
| venv paketleri | torch 2.7.0.dev, numpy 1.26.4, Pillow 11.0.0, MPS=True |
| Referans kaynak klonu | `/Users/differenterman/yonder-hud-build/comfyui-ref` |
| **ComfyUI sürümü (referans)** | **`0.24.0`** (`comfyui_version.py`), HEAD `7d4194d` 2026-06-15 |

> Not: Makinedeki `~/ComfyUI` (Aralık 2024) ve `~/comfyui` çok eski — SAM3 **yok**.
> SAM3'lü aktif çekirdek app güncellemesinde silinmiş. Bu yüzden otoriter kaynak
> olarak güncel master klonlandı ve venv ile import doğrulandı.

## 1. SAM3_Detect kaynak konumu

`comfy_extras/nodes_sam3.py` → sınıf `SAM3_Detect(io.ComfyNode)` (satır 88).
V3 şema (`define_schema`/`execute`), kayıt `SAM3Extension(ComfyExtension)` +
`comfy_entrypoint()` (satır 521-533).

Şema özeti (satır 91-113):
- inputs: `model` (MODEL), `image` (IMAGE), `conditioning` (CONDITIONING, opt),
  `bboxes` (**BOUNDING_BOX**, opt, force_input), `positive_coords`/`negative_coords`
  (STRING JSON, opt), `threshold` (FLOAT), `refine_iterations` (INT),
  `individual_masks` (BOOLEAN, default False).
- outputs: **`masks` (MASK)**, **`bboxes` (BOUNDING_BOX)**.

## 2. BOUNDING_BOX — tip beyanı

`comfy_api/latest/_io.py` satır 1283:
```python
@comfytype(io_type="BOUNDING_BOX")
class BoundingBox(ComfyTypeIO):
    class BoundingBoxDict(TypedDict):
        x: int
        y: int
        width: int
        height: int
    Type = BoundingBoxDict
    # Input default = {"x": 0, "y": 0, "width": 512, "height": 512}
```
- io_type string = **`"BOUNDING_BOX"`** (venv ile doğrulandı).
- Atomik eleman = **xywh dict** (x,y = sol-üst köşe; width,height). cxcywh veya xyxy DEĞİL.
- `.Input(force_input=, optional=)` ve `.Output()` destekli.

## 3. BOUNDING_BOX — RUNTIME yapısı (en kritik bulgu)

`SAM3_Detect.execute` (satır 116-255) çıktıyı şöyle kurar:
```python
all_bbox_dicts = []                      # outer: frame başına bir liste
for b in range(B):
    frame_bbox_dicts = []                # inner: bu frame'deki detection'lar
    for box, score in zip(kept_boxes, kept_scores):
        frame_bbox_dicts.append({
            "x": float(box[0]), "y": float(box[1]),
            "width":  float(box[2] - box[0]),
            "height": float(box[3] - box[1]),
            "score":  float(score),
        })
    all_bbox_dicts.append(frame_bbox_dicts)
return io.NodeOutput(mask_out, all_bbox_dicts)
```

**Sonuç — SAM3_Detect'in BOUNDING_BOX runtime değeri:**
```
list[ list[ dict{"x","y","width","height","score"} ] ]
  outer length = B  (frame/batch sayısı)
  inner = o frame'deki detection sayısı (boş [] olabilir)
```
- **Koordinatlar absolute pixel** (model `orig_size=(H,W)` ile çağrılır; ayrıca
  SAM3'ün bbox **input** yolu `_boxes_to_tensor` `d["x"]/W` ile normalize eder →
  girişin piksel olduğunu kanıtlar, satır 122-128).
- **`label` yoktur** (SAM3 yalnızca `score` koyar). Caption "bbox_label" modu SAM3
  için boş döner — bu beklenen davranış (brief §13). `score` her zaman vardır.
- `mediapipe` node'u aynı tipte ama `label` **ile** üretir
  (`nodes_mediapipe.py:308` → `{"x","y","width","height","label","score"}`),
  outer/inner aynı `list[list[dict]]`.

**Adapter bu üç şekli normalize etmeli** (SAM3'ün kendi giriş kodu da hepsini kabul eder):
1. Tek `dict` → tüm frame'ler için aynı
2. `list[dict]` → tüm frame'ler için aynı
3. `list[list[dict]]` → frame başına ayrı (SAM3'ün ürettiği)

Ek savunma (brief §4.3): xyxy gelirse algıla, xywh'e/xyxy'ye çevir; normalize (0..1)
gelirse W/H ile piksele çevir; ters/clamp/min-size guard uygula.

## 3.5 SAM3_TRACK_DATA — RUNTIME yapısı (BİRİNCİL tracking girişi)

> Mimari revizyon (kullanıcı, 2026-06-15): renderer'ın **birincil** tracking girişi
> custom IoU tracker değil, native persistent SAM3 track data olmalı. Öncelik:
> **(1) SAM3_TRACK_DATA → (2) BOUNDING_BOX+MASK → (3) custom IoU tracker (fallback)**.

`comfy_extras/nodes_sam3.py:258` → `SAM3TrackData = io.Custom("SAM3_TRACK_DATA")`.
`SAM3_VideoTrack` (node_id `SAM3_VideoTrack`) bunu üretir; tüketiciler:
`SAM3_TrackToMask`, `SAM3_TrackPreview`.

Üretim zinciri: `SAM3_VideoTrack.execute` → `sam3_model.forward_video(...)`
(`comfy/ldm/sam3/detector.py:563`) → `tracker.track_video_with_detection(...)`
(`comfy/ldm/sam3/tracker.py`). Dönüş (tracker.py:1791/1802) + node'un eklediği `orig_size`:

```python
SAM3_TRACK_DATA = {                       # düz Python dict
  "packed_masks": Tensor | None,          # [N_frames, N_obj, H_m, W_m//8] uint8 (bit-packed)
  "n_frames":     int,                    # N (None olsa bile frame sayısı)
  "scores":       list[float],            # uzunluk N_obj; obje başına skor; [] if None
  "orig_size":    (H, W),                 # SAM3_VideoTrack.execute satır 311 ekler
}
```

Kesin semantik (kaynaktan):
- **`packed_masks` dim 1 (N_obj) = KALICI OBJE/TRACK KİMLİĞİ.** Obje `i` tüm
  frame'lerde aynı izlenen özne; `scores[i]` onun skoru. Frame başına obje sayısı
  video genelinde `max_obj`'a **pad** edilir (tracker.py:1793-1801): bir frame'de
  görünmeyen obje all-zero (boş) mask alır → o frame'de occlusion, ama **kimlik
  index ile sabit kalır** ("track identity survives short occlusions" bedava gelir).
- **Bit-pack format** (tracker.py:194-204):
  `pack_masks: [*,H,W] bool → [*,H,W//8] uint8` (W 8'e bölünebilir).
  `unpack_masks: [*,H,W//8] uint8 → [*,H,W] bool` = `(packed.unsqueeze(-1) & bits).bool().view(...)`,
  `bits=[1,2,4,8,16,32,64,128]`. → Saf tensor işlemi. **Kendi modülümüzde
  reimplemente edilecek** (Cloud/sürüm-drift güvenliği; `comfy.ldm.sam3`'e zorunlu
  bağımlılık yok). Native `from comfy.ldm.sam3.tracker import unpack_masks` da mevcut.
- `packed_masks is None` (hiç track yok): `{"packed_masks":None,"n_frames":N,"scores":[]}`.
- Mask çözünürlüğü `H_m,W_m` tracker iç çözünürlüğü → `orig_size (H,W)`'ye
  `F.interpolate(..., size=(H,W))` ile ölçeklenir (SAM3_TrackToMask:517 gibi),
  sonra bbox mask extent'ten çıkarılır.

**Renderer track_adapter (birincil yol):** her frame `t`, her obje `i` için
`unpack(packed_masks[t,i])` → boş değilse: `Detection(frame_index=t, source_index=i,
xyxy=mask_extent, score=scores[i], source_track_id=i, mask_index=i)`. `source_track_id`
dolu olduğunda renderer'ın tracker'ı **bypass** edilir (kimlik native index'ten gelir);
deterministik stil yine `hash(seed, source_track_id)` ile. BOUNDING_BOX yolunda
`source_track_id=None` → IoU tracker devreye girer.

**Socket uyumu (doğrulandı, renk/isimden DEĞİL):** `SAM3_VideoTrack` çıktısı
`io.Custom("SAM3_TRACK_DATA").Output`. Renderer girişi `io.Custom("SAM3_TRACK_DATA").Input`
→ io_type string `"SAM3_TRACK_DATA"` birebir eşleşir → uyumlu. (nodes_sam3.py:258 ile teyit.)

## 3.6 YONDER_ANNOTATIONS — yeni semantik katman (custom type, bizim tanımımız)

> Genişletme (kullanıcı, 2026-06-15): SAM3 kimlik+mask verir ama **zengin betimsel
> prose vermez**. Detaylı label'lar (örn. "SHIRTLESS MAN WITH TATTOOS KICKING SOCCER
> BALL") ayrı, model-bağımsız bir annotation katmanından gelmeli. Renderer bunları
> **asla kendi uydurmaz** — JSON / VLM workflow / manuel / fallback template'ten alır.

Mimari = 3 bağımsız katman: **Tracking** (SAM3_TRACK_DATA → stabil kimlik) →
**Annotation** (stabil track'lere label/açıklama/attribute/child region iliştir) →
**Rendering** (PNG/procedural frame + crosshair + label strip).

Yeni custom type: **`YONDER_ANNOTATIONS`** (io_type string `"YONDER_ANNOTATIONS"`),
taşıma: ya `AnnotationSet` objesi ya da **JSON STRING** (evrensel uyumluluk). Şema v1
`yonder_hud/annotations.py`'de dataclass + validator olarak. Track kimliği **namespaced**
(`person:0`, `face:0`, `footwear:0`, `ball:0`) — çıplak index global unique değil.
Detaylı label üretimi VLM gerektirir; core renderer'a hiçbir VLM (Florence/Qwen/Moondream)
gömülmez — caption üretimi opsiyonel, composable workflow olarak dışarıda kalır (Cloud uyumu korunur).

## 4. MASK — batch yapısı

`SAM3_Detect` mask çıktısı (satır 239-254), `individual_masks`'a göre **değişir**:

- `individual_masks=False` (default):
  frame başına tek **union** mask `[H,W]` → `torch.stack` → **`mask_out: [B, H, W]`**.
  Instance ayrımı yapılamaz (sadece "bu frame'de maske olan bölge").
- `individual_masks=True`:
  frame başına `[N_obj, H, W]` → `torch.cat(dim=0)` → **`[ΣN_obj, H, W]` FLATTENED**.
  Frame sınırları tensor içinde **kodlanmaz**; frame→mask eşlemesi için bbox
  dict'lerinin frame başına sayısı kullanılmalı (saf `person` text-prompt'ta
  mask sırası ile bbox sırası 1:1 hizalıdır — ikisi de `kept` sırasından gelir).
- Detection yoksa: `individual` → `[0,H,W]`, union → `[H,W]` (sıfır).

MASK io_type = `"MASK"`, `Type=torch.Tensor`. Değerler float, binary 0.0/1.0
(`.float()` of bool). Genel MASK konvansiyonu `[B,H,W]`.

> Renderer politikası: `individual_masks=True` + frame başına bbox sayısı = en
> güvenilir instance hizalaması. Aksi halde mask yalnızca fallback bbox üretimi
> (Mask→BBox) için kullanılır, kişi ayrımı bbox'tan gelir.

## 5. IMAGE — batch yapısı

io_type=`"IMAGE"`, `Type=torch.Tensor`. SAM3 `B,H,W,C = image.shape` (satır 117) →
**`[B, H, W, C]`**, float **0.0..1.0**. C=3 (RGB) veya 4 (RGBA; `image[...,:3]` ile
RGB alınır). İşlemede `movedim(-1,1)` → `[B,C,H,W]`. Renderer giriş/çıkış aynı
`[B,H,W,3]` float32 olmalı (brief §17).

## 6. Video frame / audio / fps veri akışı (native, 3rd-party yok)

`comfy_extras/nodes_video.py` (ComfyUI 0.24.0) native node'ları:
| node_id | giriş | çıkış |
|---------|-------|-------|
| `LoadVideo` | dosya | VIDEO |
| `GetVideoComponents` | VIDEO | **images (IMAGE), audio (AUDIO), fps (FLOAT)**, +bit_depth (satır 165-188) |
| `CreateVideo` | images (IMAGE), fps (FLOAT), audio (AUDIO, opt) | VIDEO (satır 127-160) |
| `SaveVideo` | VIDEO | dosya |
| `SaveWEBM` | images, codec, fps | dosya |

`GetVideoComponents.execute` → `io.NodeOutput(components.images, components.audio,
float(components.frame_rate), video.get_bit_depth())`.
`VideoComponents` = `{images, audio, frame_rate: Fraction}` (`comfy_api/.../_util/video_types.py`).

**MVP veri akışı (kanıtlanmış, native, bridge yok):**
```
LoadVideo → VIDEO
   → GetVideoComponents → images(IMAGE) + audio(AUDIO) + fps(FLOAT)
images → SAM3 Detect → masks(MASK) + bboxes(BOUNDING_BOX)
images + masks + bboxes + asset PNG'leri → Yonder HUD Renderer → images(IMAGE)
Yonder images + fps + audio → CreateVideo → VIDEO → SaveVideo
```
> Brief diyagramındaki tek "Video Segment (SAM3)" kutusu audio/fps **vermez**;
> audio+fps `GetVideoComponents`'ten gelir. SAM3 yalnızca IMAGE alır, MASK+BBOX verir.

## 7. VIDEO tipi (2. aşama wrapper için)

io_type=`"VIDEO"`, `Type=VideoInput` (düz dict **DEĞİL** — soyut wrapper sınıf,
`comfy_api/latest/_input/video_types.py`). Stage-2 `Yonder HUD Video Renderer`:
`video.get_components()` → images/audio/frame_rate al → images'i işle →
`VideoFromComponents(VideoComponents(images=..., audio=..., frame_rate=...))` ile
geri sar. MVP'de VIDEO **uydurulmaz**; IMAGE batch + GetVideoComponents/CreateVideo yolu kullanılır.

## 8. Node registration yöntemi — V3 (birincil) + V1 fallback

Loader (`nodes.py` 2241-2281): `if NODE_CLASS_MAPPINGS: ... elif comfy_entrypoint:`.
→ İkisi birden export edilirse **NODE_CLASS_MAPPINGS önceliklidir**, comfy_entrypoint yok sayılır.

Karar:
- **Birincil: V3** (`comfy_api.latest`: `io.ComfyNode` + `define_schema` + `execute` +
  `ComfyExtension` + `comfy_entrypoint`) — SAM3'ün kendisi V3, target 0.24.0 V3 destekli.
- **Fallback: V1** — `__init__.py` içinde `comfy_api.latest` import edilemezse
  (eski ComfyUI) `NODE_CLASS_MAPPINGS` (INPUT_TYPES shim) üret. Loader önceliği
  nedeniyle V1 mapping yalnızca V3 yokken tanımlanır. **İş mantığı `yonder_hud/`
  içinde schema-agnostik** kalır; iki ince adapter, kod duplikasyonu yok (brief §20).

Doğrulanan V3 input API'leri (SAM3 + _io.py):
`io.Image.Input(id, optional=)`, `io.Mask.Input(...)`,
`io.BoundingBox.Input(id, optional=, force_input=)`,
`io.Int.Input(id, default=, min=, max=, step=)`, `io.Float.Input(...)`,
`io.Boolean.Input(...)`, `io.String.Input(..., multiline=)`,
`io.Combo.Input(id, options=[...], default=)`,
`io.Schema(node_id=, display_name=, category=, inputs=[], outputs=[], is_output_node=, description=, search_aliases=[])`,
`io.NodeOutput(out1, out2, ...)`, progress: `comfy.utils.ProgressBar(total)` + `.update(1)`.

## 9. Önerilen repository file structure

Brief §5 yapısı korunur; kayıt katmanı V3/V1 ayrımıyla:
```
ComfyUI-YonderHUD/
├── __init__.py            # V3 comfy_entrypoint; comfy_api yoksa V1 NODE_CLASS_MAPPINGS fallback
├── pyproject.toml, README.md, LICENSE, requirements.txt   # Pillow, numpy (torch ComfyUI'dan)
├── nodes/                 # İNCE schema katmanı (V3 + v1_compat)
│   ├── hud_renderer.py inspect_bbox.py mask_to_bbox.py asset_pack.py video_wrapper.py v1_compat.py
├── yonder_hud/            # schema-AGNOSTIK iş mantığı (test edilebilir, torch/Comfy bağımsız mümkün olduğunca)
│   ├── types.py geometry.py alpha.py palette.py text_render.py procedural_hud.py
│   ├── bbox_adapter.py mask_adapter.py tracker.py compositor.py captions.py assets.py utils.py
├── web/js/yonder_hud.js   # opsiyonel UX (backend bağımsız çalışır)
├── examples/              # sam3_hud_image_batch_workflow.json, sam3_hud_video_workflow.json, README.md
├── tests/                 # test_bbox_adapter/mask_adapter/tracking/alpha/compositor/captions + fixtures + test_integration_sam3
└── docs/                  # runtime_findings.md(✓) architecture.md node_reference.md troubleshooting.md
```

## 10. Integration test edilebilirliği

- venv ile `from comfy_api.latest import io` **çalışıyor** → node'lar headless register & execute edilebilir.
- Gerçek SAM3 **inference** için checkpoint yok (`models/` altında SAM3 ağırlığı yok) →
  GPU end-to-end şu an mümkün değil. Bunun yerine: SAM3_Detect'in **birebir** dict-üretim
  kodu (bu doküman §3) ile fixture üretilir + gerçek `SAM3_Detect.define_schema()` output
  tipinin `BOUNDING_BOX` olduğu assert edilir → adapter gerçek tip kontratına bağlanır.
- Bilinen sınır: canlı SAM3 ağırlık indirildiğinde kullanıcı example workflow'u koşar
  (README'de adımlar). fps/audio passthrough `GetVideoComponents`+`CreateVideo` ile native.
