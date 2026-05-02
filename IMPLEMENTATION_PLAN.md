# Psaltica OCR — Implementation Plan

## Context

`/Users/nadcost/psaltica-ocr` is an empty workspace, just initialized with `bd` task tracking. The goal is to build a **local OMR (Optical Music Recognition) pipeline** for printed Byzantine chant scores that outputs cluster strings consumable by the existing Psaltica Praxis app at `/Users/nadcost/psaltica-praxis`.

Why this is feasible now: the user owns a private corpus of Byzantine PDFs (training material), and the Psaltica Praxis app already encodes the entire symbol vocabulary canonically across `app/core/toolbars.ts`, `app/core/keySignatures.ts`, and `app/core/music/actionMap.ts`. The OCR project does **not** need to invent a notation; it needs to recognize visual glyphs and emit the app's existing cluster format.

Intended outcome: a local Python pipeline that converts a clean printed-PDF page into a Psaltica composition JSON, with a review/correction UI to drive an iterative training loop. Mobile/on-device deployment is a later phase out of scope here.

---

## Architecture (data flow)

```
PDFs ──▶ Renderer ──▶ Page images (300-400 dpi)
                         │
                         ▼
                  Chant-region masking (lyrics rows dropped)
                         │
                         ▼
                  Annotation (Label Studio)
                         │
                         ▼
                  Labeled dataset (YOLO format)
                         │
                         ▼
                  Detector (YOLOv11) ──▶ Glyph boxes + classes
                                           │
                                           ▼
                          Atomic-token pass (key sigs, rests)
                                           │
                                           ▼
                          Cluster Assembly (deterministic Python)
                                           │
                                           ▼
                          Strict OCR Validator  +  clusterParser.ts port (sanity)
                                           │
                                           ▼
                          Composition JSON  ──▶  Review UI (Streamlit)
                                                       │
                                                       ▼
                          4 correction tracks ──▶ retrain / rule-tune
```

**Hard separation of concerns:** ML detects visual symbols only. Byzantine grammar (atomic tokens, cluster grouping, variant selection, segment direction, validity) lives in deterministic post-processing code we control.

---

## Repo structure (target)

```
psaltica-ocr/
  config/
    classes.yaml             # OCR class taxonomy (codegen'd)
    symbol_map.json          # full canonical map (codegen'd from app)
  data/
    pdfs/                    # source books (gitignored, large)
    pages/<book>/<page>.png  # rendered images (gitignored)
    annotations/             # raw Label Studio exports (gitignored)
    datasets/                # YOLO-formatted train/val/test (gitignored)
    corrections/<page>/      # review-UI outputs (4 tracks, see Phase 5)
  psaltica_ocr/              # Python package
    __init__.py
    rendering.py             # PDF → page images, chant/lyric masking
    detection.py             # YOLO inference wrapper
    atomic_tokens.py         # key-signature & rest token recognition
    cluster_assembly.py      # spatial rules → cluster strings
    cluster_parser.py        # Python port of clusterParser.ts (sanity only)
    ocr_validator.py         # strict validator (THE quality gate)
    symbol_map.py            # loads config/symbol_map.json
    schemas.py               # pydantic models for JSON output
    io_psaltica.py           # composition JSON writer matching app format
  tools/
    sync_symbol_map.py       # Python wrapper that runs:
    _extract_symbol_map.ts   # ts-node script: imports app modules, writes JSON
    render_pdfs.py
    export_label_studio_config.py
    import_labels.py         # Label Studio JSON → YOLO format
    train.py                 # ultralytics wrapper
    eval.py                  # mAP + cluster-level accuracy
    infer_page.py            # page → composition JSON
  review_ui/
    app.py                   # Streamlit app
  models/
    runs/                    # YOLO training runs (gitignored)
    exported/<version>/      # versioned bundled model + class map
  tests/
    test_atomic_tokens.py
    test_cluster_assembly.py
    test_cluster_parser.py
    test_ocr_validator.py
    test_symbol_map.py
  pyproject.toml             # uv-managed deps
  .python-version            # 3.11+
  README.md
```

---

## Phased roadmap

### Phase 0 — Bootstrap (≈ 1 week)

**Deliverables:**
- `pyproject.toml` with `uv` lockfile; deps: `pdf2image`, `pillow`, `opencv-python-headless`, `numpy`, `ultralytics`, `pandas`, `pydantic`, `pyyaml`, `pytest`, `streamlit`.
- System deps documented: `brew install poppler node` (node needed for the symbol-map extractor).
- **Symbol-map codegen** — read directly from app TypeScript modules, NOT from CSV:
  - `tools/_extract_symbol_map.ts`: a `tsx` script (NOT `ts-node`) executed with cwd set to `/Users/nadcost/psaltica-praxis` so the app's relative imports resolve predictably. It imports the live app modules:
    - `app/core/toolbars.ts` → modesToolbar, modulationToolbar, neumeToolbar, gorgonToolbar, issonToolbar (count asserted at extraction time)
    - `app/core/keySignatures.ts` → RAW_KEY_SIGNATURES (count asserted at extraction time — do NOT hard-code 49; let the count test fail if the app adds or removes entries)
    - `app/core/music/actionMap.ts` → ACTION_CHAR_MAP + REACT/LEGACY/ALL_SEQUENCE_TO_ACTION (count asserted at extraction time)
    - Length classification arrays from toolbars.ts (`longNeumes`, `shortNeumes`, `extraShortNeumes`, `heavyTopNeumes`)
    - `KLASMA_PLACEMENT_BY_ICON` from clusterCatalog.ts
  - Writes `config/symbol_map.json` with this schema per entry:
    ```json
    {
      "icon": "Oligon",
      "label": "Oligon",
      "group": "neume" | "gorgon" | "modulation" | "isson" | "mode" | "key_signature" | "rest" | "ornament",
      "role": "base" | "modifier" | "key_signature" | "rest" | "ornament",
      "variants": {"insert": "1", "short": null, "long": null, ...},
      "insert": "1",                           // primary char or composite
      "isBase": true,
      "isModifier": false,
      "isKeySignature": false,
      "keyId": null,                            // populated when isKeySignature
      "category": null,                         // "Diatonic"|"Enharmonic"|"Soft Chromatic"|"Hard Chromatic"
      "basePitch": null,
      "length": "long" | "short" | "extraShort" | "normal",   // for bases
      "heavyTop": false,                                       // for bases
      "klasmaPlacement": "topCenter" | "topLeft" | "topRight" | "bottomCenter",  // for bases
      "legacyChars": {"short": ":", "long": ";"},              // from CSV/actionMap, fallback only
      "reactChars":  {"short": ":", "long": ";"}
    }
    ```
  - `tools/sync_symbol_map.py`: thin Python wrapper that shells out to the TS script, then validates the output and emits `config/classes.yaml` (the OCR class list — see "Class taxonomy" below).
- The CSV (`legacy-react-char-map.csv`) is consulted **only** as a fallback for `legacyChars` populating where actionMap doesn't have a legacy variant. It is NEVER the canonical taxonomy source.
- `psaltica_ocr/symbol_map.py` loads + validates the JSON via pydantic.
- Drift detection: `sync_symbol_map.py --check` mode that re-runs extraction and diffs against the committed `symbol_map.json` — fails CI on drift.
- Tests: every char in `symbol_map.json` is unique within its `(group, variant)` scope; every `RAW_KEY_SIGNATURES.insert` decomposes into known chars; every `ACTION_CHAR_MAP` icon appears in exactly one toolbar OR is a key signature OR is explicitly orphan; **the extractor records the live counts** of toolbar items, key signatures, and action-map entries into `symbol_map.json._meta` and a separate count test asserts those counts — drift fails CI loudly rather than silently mis-classing OCR output.

**Decision (revised, was high finding #1):** Source of truth is `toolbars.ts` + `keySignatures.ts` + `actionMap.ts`, NOT the CSV. CSV labels (e.g. `Modulation1`) are stale relative to current app labels (`ModulationDiatonicPa`), and CSV does not include the expanded 49-entry key-signature catalog. Reading from the live TS modules guarantees the OCR taxonomy never drifts from the app.

### Phase 1 — Dataset pipeline (≈ 1-2 weeks)

**Deliverables:**
- `tools/render_pdfs.py`: PDF → 400 dpi PNGs, deterministic naming `data/pages/<book_id>/page_<NNNN>.png`, manifest CSV with hash + dpi + page dims.
- `psaltica_ocr/rendering.py`: shared rendering primitives (deskew, binarize, dewarp helpers using OpenCV).
- **Chant-region segmentation (in scope, was finding #6):** a `mask_lyrics(image)` function that returns a binary mask of probable chant rows. Two implementations:
  - v0 (heuristic): horizontal projection profile + row-height clustering. Chant rows have higher glyph variance and taller bbox heights than text rows.
  - v1 (learned): if v0 is unreliable, add `lyrics_text` as a YOLO class during Phase 2 annotation and drop those boxes after detection.
  - Either way, lyrics rows are masked out before annotation/training to prevent the detector from learning lyrics characters as false-positive neume classes.
- Initial corpus audit: pick 2-3 books with the cleanest print + most common style, defer scans/photos.
- `bd-mpj`: closed when 200-500 pages are rendered, masked, and manifested.

### Phase 2 — Annotation workflow (≈ 1-2 weeks)

**Deliverables:**
- `tools/export_label_studio_config.py`: emits Label Studio XML config from `config/classes.yaml`.
- 50-page PoC annotation pass on one book: bounding boxes + class for every chant glyph in unmasked regions.
- `tools/import_labels.py`: Label Studio JSON export → YOLO `.txt` per image + `dataset.yaml`.
- Annotation guidelines stored via `bd remember`: tight boxes; collapse variant by default (see below); annotate complete key-signature glyphs as single boxes (NOT as base+modifier); annotate rests as their own class.
- **Seed gold compositions early.** For 5-10 representative pages from the annotation set (choose pages that exercise base+modifier clusters, key signatures, and rests), write `data/corrections/<page>/expected_composition.json` by hand. Format: `{"segments": [{"composition": "vV1S2a...", "direction": "ltr"}]}` — same shape as the pipeline's `_ocr`-stripped output. These gold files are the cluster-level accuracy target from Phase 3 onward, well before the review UI exists. Detection mAP alone will not tell you whether the end-to-end pipeline is useful.
- `bd-mgr`: closed when 50 pages are annotated, exported, AND ≥5 gold `expected_composition.json` files are written.

**Decision (revised, was finding #4 — variant taxonomy):**
The class taxonomy is decided by **rendered shape + spatial role**, not by codepoint distinctness:
- Default rule: collapse all variants of an icon into a single class (`Gorgon`, not `Gorgon.short`/`Gorgon.long`).
- Split rule: split a class only after observing that two variants render as visibly distinct shapes AND occupy distinct spatial slots in the printed score. The codepoint table in `actionMap.ts` is informative but not decisive.
- Workflow: train v0 with all-variants-collapsed. Inspect per-class confusion matrix. If specific variants confuse with each other or with unrelated classes, split. Iterate.
- Variant **selection** in the output (which char to emit) is always deterministic and lives in cluster assembly, driven by the base neume's `length`/`heavyTop`/`klasmaPlacement` from `symbol_map.json`. Never use the model's class output as the variant authority.

Class groups in `classes.yaml` reflect the assembly architecture:
- `base_neume.*` (~24 classes) — opens a cluster
- `modifier_gorgon.*` (~16) — attaches to base
- `modifier_modulation.*` (~19) — attaches to base or stands between
- `modifier_isson.*` (~8) — attaches to base
- `key_signature.*` (count from `symbol_map.json._meta.key_signature_count` — do not hard-code; see Phase 0)
- `rest.*` (TBD count) — atomic
- `ornament.*` (apli, dipli, etc., from actionMap)
- `lyrics_text` (only if v1 lyrics-masking strategy is chosen)

### Phase 3 — Detector v0 (≈ 2 weeks)

**Deliverables:**
- `tools/train.py`: ultralytics CLI wrapper, `imgsz=1280`, `model=yolo11s.pt`, `epochs=100`, `patience=20`. Training runs go to `models/runs/<timestamp>/`.
- `tools/eval.py`: per-class precision/recall/mAP@0.5 + confusion matrix; outputs CSV for tracking across versions.
- `tools/infer_page.py`: load model, run on one page, dump detection JSON `{boxes:[{xyxy, class, conf}]}`.
- Target: > 0.85 mAP@0.5 on validation set (symbol-level).
- **Cluster-level smoke test (informational, not a gate here):** run `infer_page.py` + assembly + `ocr_validator` on the 5-10 gold pages from Phase 2 and report cluster-level accuracy against `expected_composition.json`. This number — not mAP — drives Phase 4 priorities and reveals whether the pipeline is actually useful end-to-end.
- `bd-50v`: closed when v0 trained, eval report committed, and the gold-set cluster-level accuracy number is recorded.

### Phase 4 — Cluster assembly + strict validation (≈ 3-4 weeks) — **the hard musical part**

This phase is restructured around **multi-pass assembly** with separate paths per token type, and a **strict validator** as the quality gate (was findings #2 and #3).

**Pass A — Atomic token recognition (`psaltica_ocr/atomic_tokens.py`):**
Run BEFORE neume clustering. Recognizes glyphs that are atomic in the cluster grammar even if visually composite:
1. **Key signatures.** All 49 entries from `RAW_KEY_SIGNATURES` are atomic tokens with composite multi-char inserts (2-4 chars). The detector outputs them as a single `key_signature.<icon>` class; the assembler emits the entry's `insert` string verbatim. Grouping into a key signature must precede neume clustering — a key sig at the start of a row opens a segment, a mid-row key sig (`role: "midOnly"`) is inserted between clusters.
2. **Rests / silences.** Detected as atomic class instances; no modifiers attach.
3. **Standalone ornaments / fthora-only markers** that the app treats as non-base atomic glyphs.

**Pass B — Base-neume clustering (`psaltica_ocr/cluster_assembly.py`):**
Operates only on detections NOT consumed by Pass A.
1. **Row segmentation**: cluster remaining detections into chant rows by y-centroid (DBSCAN or 1D histogram peaks).
2. **Reading order**: sort each row left-to-right (encoding direction is always L→R; render direction is per-segment metadata only — confirmed at `tabManager.ts:14`).
3. **Cluster grouping**: walk left to right; a `base_neume.*` opens a cluster; subsequent `modifier_*` glyphs within proximity radius (above/below/within next-base x-gap) attach to it.
4. **Variant resolution**: pick `short`/`long`/`extraShort`/`under` per modifier using `BaseCharInfo.length` + `heavyTop` rules from `clusterCatalog.ts:66-83`, mirrored in Python via `symbol_map.json` fields.
5. **Char emission**: look up `(icon, variant)` in `symbol_map.json` → char. Concatenate `base_char + modifier_chars` per cluster.

**Pass C — Composition assembly:**
Interleave Pass A tokens and Pass B clusters in row reading order. Each row becomes a `Segment` with `composition` string. Multi-row pages produce multi-segment compositions. Lyrics text (if detected) attaches as `lyrics: string[]` on the corresponding segment if a lyric-row sits directly below a chant row; otherwise empty.

**Strict OCR validator (`psaltica_ocr/ocr_validator.py`) — THE quality gate:**
Replaces "parseable composition" with "strictly valid composition". The Python port of `clusterParser.ts` is kept ONLY as a sanity check (does the editor accept this string?), because that parser is intentionally permissive — it classifies unknown chars as `trailing` rather than rejecting them, so a pure round-trip cannot detect invalid OCR output.

The strict validator enforces, per cluster/token in the composition stream:
1. **Shape**: every token is exactly one of:
   - a known `base_neume` char optionally followed by zero-or-more known `modifier_*` chars (in any order, but each modifier appearing at most N times where N comes from the icon's spec);
   - a known `key_signature` composite insert (matched as a whole multi-char sequence);
   - a known `rest` / `ornament` atomic char.
2. **No trailing**: zero unknown / unclassified chars after a token.
3. **Legal modifier attachment**: modifier groups have rules expressed as data in `symbol_map.json`:
   - `modifier_gorgon` may attach only to `base_neume` (not to keys or rests).
   - `modifier_isson`: **no per-base attachment restriction in v0.** The toolbar exposes ison variants as modifiers, but which bases they may attach to is not documented in the app code. Over-constraining the validator before app behavior confirms the allowed set would reject valid OCR output. Add a per-base rule only once observed app behavior justifies it.
   - `modifier_modulation` (fthoras) may appear before/after a base depending on the entry's metadata.
   - At most one of each mutually-exclusive modifier per cluster (no two gorgons on one base).
4. **Segment-level**: if a segment starts with a key signature, role must be `"segmentStart"`; mid-row key signatures must be `role: "midOnly"`.
5. **Returns** a list of `ValidationError(segmentIndex, position, code, message)` — empty list = strictly valid.

Acceptance gate for Phase 4: `ocr_validator.validate(composition_json)` returns `[]` for ≥80% of held-out test pages, AND `cluster_parser.parseCluster()` round-trips the same strings without producing `trailing[]` chars.

**Output (`psaltica_ocr/io_psaltica.py`)** matches the app's `applyDraft` shape (`ComposerScreen.tsx:8087-8145`):
```json
{
  "id": "imported_<book>_p<page>",
  "tempo": 120,
  "segments": [
    {"id": "...", "direction": "ltr", "composition": "vV1S2a...", "lyrics": []}
  ],
  "_ocr": {
    "modelVersion": "yolo11s-v0.3",
    "sourceImage": "data/pages/book_001/page_0007.png",
    "warnings": [...],
    "validatorErrors": []
  }
}
```

`bd-8xr`: closed when end-to-end inference produces strictly-valid composition JSON for the 50-page set with ≥80% pages clean.

### Phase 5 — Local review UI (≈ 2 weeks)

**Deliverables:**
- `review_ui/app.py` Streamlit app:
  - Left pane: source page image with detection overlay (color = confidence, click-to-edit class).
  - Right pane: assembled composition rendered as text (the Byzantine font from psaltica-praxis, or icon names + char codes as fallback).
  - Validator-error inspector: errors from `ocr_validator` shown inline at the offending position.
- **Four parallel correction tracks (was finding #5)**, written under `data/corrections/<page>/`:
  1. `detections.yolo` — box/class corrections. Feeds detector retraining.
  2. `cluster_overrides.json` — manual cluster grouping/variant overrides (e.g. "this gorgon belongs to the next base, not this one"). Feeds cluster-assembly rule tuning, NOT the detector.
  3. `reading_order.json` — manual row order / per-row reading direction overrides. Feeds the row-segmentation algorithm.
  4. `expected_composition.json` — gold composition string for this page. **Seeded manually for 5-10 pages in Phase 2**, then expanded here as the review UI generates more. The cluster-level accuracy metric (Phase 4 gate) compares assembled output to this.
- Each correction artifact is independently consumed by its respective subsystem, so a grouping fix doesn't pollute detector training and a missed-box fix doesn't pollute the assembly rule audit.
- Out of scope: live re-inference after edits; replay-from-history.

### Phase 6 — Iteration loop

Retrain on corrected `detections.yolo` → v1, v2, … Each model gets a directory under `models/exported/<semver>/` with: `weights.pt`, `classes.yaml` snapshot, `eval_report.csv`, `model_card.md`. Cluster-assembly and validator rules are versioned alongside in `psaltica_ocr/`. Model version + rule-set version are embedded in inference output JSON for traceability.

TFLite export (Phase 6+) only when v1 hits target accuracy. Not before.

### Phase 7 — App integration (out of scope here)

When OCR output is reliable, an "Import recognized score" path is added in `psaltica-praxis` (separate repo). The app reads the same JSON shape it already imports for drafts; only the entry point is new. **This plan does not touch psaltica-praxis.**

---

## Stack & tooling decisions

| Concern | Choice | Rationale |
|---|---|---|
| Python | 3.11+ | Modern typing, ultralytics support |
| Pkg manager | `uv` | Fast, reproducible lockfile |
| TS extractor runtime | `tsx` (run from `psaltica-praxis` cwd) | Resolves app's relative imports predictably; single committed runtime, no fallback |
| PDF render | `pdf2image` + poppler | Standard, deterministic |
| Image ops | OpenCV + Pillow | Industry-standard |
| Detector | YOLOv11s (ultralytics) | Best accessibility/perf trade-off; swap to RT-DETR later if needed |
| Annotation | Label Studio (local Docker or pip) | XML config = codegen friendly |
| Review UI | Streamlit | Fastest path to a working tool for solo dev |
| Schema | pydantic v2 | Validates JSON shape, strong types |
| Tests | pytest | Standard |

All large artifacts (PDFs, page images, annotations, model weights) are gitignored. Only code, configs, and small fixtures live in git.

---

## Critical design decisions (recap, post-review)

1. **Symbol map source = live app TypeScript, not CSV.** `tools/_extract_symbol_map.ts` imports `toolbars.ts` + `keySignatures.ts` + `actionMap.ts` and emits `config/symbol_map.json` with full schema (icon, label, group, role, variants, insert, isBase, isModifier, isKeySignature, keyId, category, length, heavyTop, klasmaPlacement, legacyChars, reactChars). CSV is fallback-only. Drift detection in CI.
2. **Strict OCR validator is the quality gate, not clusterParser.ts.** clusterParser is permissive (unknowns → `trailing[]`). The strict validator rejects unknown chars, illegal modifier attachments, role mismatches, and unclassified trailing. clusterParser port is kept as a sanity check only.
3. **Key signatures and rests are atomic tokens, recognized BEFORE neume clustering.** Pass A consumes them; Pass B does base+modifier clustering only on remaining detections; Pass C interleaves into reading order. No more "base opens cluster" universal assumption.
4. **Class taxonomy is collapsed by default and split empirically.** Visual shape + spatial role decide split/merge — not codepoint distinctness. v0 trains all-variants-collapsed; splits emerge from confusion-matrix analysis.
5. **Direction is metadata, not encoding.** Encoding is always L→R char order; per-segment `direction: "ltr"|"rtl"` is set by config or detected later.
6. **Chant/lyrics segmentation is in scope (v0 heuristic, v1 learned class).** Lyrics rows are masked out before training so the detector doesn't learn lyrics characters as false positives.
7. **Review corrections are split into 4 tracks** (detections, cluster_overrides, reading_order, expected_composition) so each correction reaches the right subsystem.
8. **No edits to psaltica-praxis.** OCR project is fully standalone; app integration is a later, separate effort.

---

## Verification

End-to-end success of Phase 4 means:
```bash
uv run tools/infer_page.py data/pages/book_001/page_0007.png \
  --model models/exported/v0.1/weights.pt \
  --out out/page_0007.json
uv run python -m psaltica_ocr.ocr_validator out/page_0007.json   # ≥80% pages: zero errors
uv run python -m psaltica_ocr.cluster_parser out/page_0007.json  # round-trip clean (no trailing[])
```
And the resulting JSON `segments[].composition` strings, when pasted into the Psaltica Praxis composer (manual import for now), render as the expected musical line.

Per-phase success metrics:
- Phase 0: `sync_symbol_map.py --check` is clean; symbol_map.json covers all toolbar items + all 49 key signatures + all 83 actionMap entries.
- Phase 1: ≥ 200 manifested pages, sample audited visually, lyrics-mask precision ≥ 90% on a 10-page audit set.
- Phase 2: ≥ 50 pages annotated, dataset YAML loads in ultralytics.
- Phase 3: mAP@0.5 ≥ 0.85 on val split; cluster-level accuracy on the 5-10 Phase 2 gold pages recorded (informational, no fixed gate).
- Phase 4: **strict-validator passes on ≥ 80% of held-out test pages**; cluster-level accuracy ≥ 80% against `expected_composition.json` gold (cluster correct = base + modifiers + variant all match ground truth string).
- Phase 5: review UI roundtrips a correction in any of the 4 tracks → next train run / next assembly run picks it up correctly.

Run `pytest` after each phase. Run `bd ready` between phases to find next work.

---

## Risks

- **Annotation throughput is the bottleneck.** 50 pages × ~200 glyphs = 10k bounding boxes minimum. Mitigation: build a v0 model on 20 pages, use it to pre-annotate the next 30 (active learning).
- **Variant ambiguity.** Some modifier variants are visually identical; if the detector confuses them, cluster assembly's spatial rules MUST be the tiebreaker — never rely on the model's class output for variants.
- **Multi-row pages.** Page segmentation into rows is not trivial; if DBSCAN fails on tightly-spaced rows, fall back to projection profile + valley detection.
- **Symbol-map drift.** The app evolves. `sync_symbol_map.py --check` runs in CI to fail loudly when the live TS modules diverge from the committed `symbol_map.json`.
- **clusterParser permissiveness leaking through.** Risk mitigated by making `ocr_validator` the gate; clusterParser is sanity-only.
- **Lyrics masking false negatives.** A lyrics row mistakenly classified as chant pollutes detector training. Mitigation: visual audit at Phase 1 + ability to switch from heuristic (v0) to learned `lyrics_text` class (v1).

---

## What this plan does NOT do

- Train on photos or handwritten notation.
- Ship anything to the mobile app — no edits to `/Users/nadcost/psaltica-praxis`.
- TFLite/Core ML export, on-device inference, or React Native integration.
- Lyrics text OCR (only chant-region masking).
- Multi-language lyric handling (Greek/Arabic/English).

These are intentionally deferred until the printed-page pipeline is reliable.
