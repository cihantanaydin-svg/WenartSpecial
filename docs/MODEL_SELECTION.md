# ArchRender: Model Selection (one ADR per role)

Status: Phase 0, proposed · Evidence checked **2026-09-30**.

## How the evidence was gathered (read this first)
- This dev container's network policy blocks `huggingface.co` (and its mirrors), `modelscope.cn`
  and the vendor sites. Evidence therefore comes from:
  - **LICENSE/README files read directly** from official GitHub repos (`raw.githubusercontent.com`).
  - **Hugging Face model-card text seen through web-search snippets**, marked *(snippet)*.
  - **PyPI metadata**.
  - **`vllm-project/recipes`**, the source of `min_vllm_version`, `vram_minimum_gb` and repo ids.
  - **diffusers source** at commit `a12c38a`, plus the wheels 0.35.0–0.40.0.
- **Nothing is trusted blindly at runtime.** `scripts/pin_models.py`, run on the pod or any machine
  with Hub access, resolves each registry entry to an exact commit and per-file SHA256 and writes
  `configs/models.lock.yaml`. It also snapshots the LICENSE file next to the weights. At download,
  `download_models.py` compares the Hub's `cardData.license` and gating flag with the registry and
  **refuses on mismatch**.
- Exact revisions and SHA256s cannot be pinned from this container. They are produced by
  `pin_models.py` in Phase 1/2 and committed, and CI fails if an enabled entry has no lock.

## License policy applied (LicenseGate)

| Class | Examples | Gate behaviour |
|---|---|---|
| `permissive` | Apache-2.0, MIT, BSD | Allowed |
| `conditional` | SAM License, DINOv3 License, CreativeML OpenRAIL++-M, CC BY 4.0 datasets | Allowed **only if** the terms id is listed in `deployment.yaml: accepted_license_terms` (owner decision). Obligations are printed into `THIRD_PARTY_LICENSES.md`. |
| `blocked` | Any non-commercial/research-only clause; territorial exclusions; MAU or revenue caps; API-only; AGPL in the network service; NC transitive dependencies (text encoders, VAEs, rasterizers) | Never loaded. For FLUX `[dev]`-class models, override only with a configured commercial license file (hash-checked). |

Jurisdictions default to the conservative placeholder in ADR-S10 until the owner fills in §0.

## Runtime versions observed (PyPI, 2026-09-30)
- vLLM 0.30.0 (2026-09-22)
- transformers 5.17.0
- diffusers 0.40.0 (2026-08-20)
- xgrammar 0.2.8
- Structured output in vLLM uses `response_format={"type":"json_schema"}` / `structured_outputs`. The
  legacy `guided_json` was **removed in v0.12.0** (vLLM `docs/features/structured_outputs.md`).
- Reasoning models need thinking disabled, or `--structured-outputs-config.enable_in_reasoning=True`,
  for JSON.
- FP8 W8A8 runs natively on Ada/Hopper/Blackwell. Ampere (A100/A6000) gets weight-only FP8 via
  Marlin (memory savings, no speed-up) (vLLM quantization docs).

---

## ADR-M01: General VLM (page classification, overlay verification, on-demand coordinate hints, brief extraction, layout proposals, QA judge)

| Candidate | License (evidence) | Gated | Size / VRAM | Runtime | Quality evidence | Verdict |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.6-27B` (+ `Qwen/Qwen3.6-27B-FP8`) | Apache-2.0: FP8 card "30.9 GB … Apache 2.0 license" *(snippet)*; Qwen3.8 GitHub README lists "2026-04-22: Qwen3.6-27B" | No | 27B dense; FP8 recipe min **33 GB**; BF16 min 65 GB | vLLM ≥ 0.17 (recipe), card recommends ≥ 0.19 | MMMU 82.9, MMMU-Pro 75.8 *(secondary)*; verified on 8 hardware targets in vLLM recipes | **Primary** |
| `Qwen/Qwen3.8-27B` (+ `-FP8`) | Apache-2.0: "ships under Apache 2.0 … native image and video input … 262,144-token context" *(snippet)*; GitHub README "2026-08-14: Qwen3.8-27B" | No | FP8 30.9 GB, recipe min 38 GB | vLLM ≥ 0.17; recipe verified **text serving only** | OmniDocBench1.5 91.1 *(aggregator)* | **Challenger.** Promoted by config if it beats the primary on our Phase-2 eval (classification F1, overlay-verification accuracy, judge accuracy on fault sets) |
| `Qwen/Qwen3.6-35B-A3B-FP8` | Apache-2.0 *(snippets)* | No | 35B MoE/3B active; FP8 min 42 GB | vLLM ≥ 0.17 | OmniDocBench1.5 89.9 *(snippet)* | Alternative for high-throughput bulk classification on ≥ 80 GB |
| `Qwen/Qwen3.5-4B` / `-9B` | Apache-2.0 *(snippet)* | No | 10 GB / 22 GB BF16 | vLLM ≥ 0.17 | Qwen reports they beat same-size Qwen3-VL | Optional cheap checker. Supersedes Qwen3-VL-2B/4B/8B. |
| Llama 4 Scout/Maverick | Llama 4 Community License: > 700M MAU clause; USE_POLICY withholds multimodal rights from EU-based companies (LICENSE + USE_POLICY read on GitHub) | Yes | FP8 131 GB | — | — | **Blocked** |
| Qwen3.8-2.4T-A95B | "Qwen3.8-Max License": > $50M revenue AI-service clause *(secondary)* | — | Far beyond 1 GPU | — | — | **Blocked** |
| InternVL3.5, GLM-4.6V-Flash, MiniCPM-V-4.6, Molmo2-8B, Kimi-VL | Code licenses MIT/Apache; InternVL weight license UNVERIFIED | — | — | — | — | Not selected (weaker or unverified); Molmo2 noted for pointing tasks |

**Decision.**
- Primary is `Qwen/Qwen3.6-27B-FP8` on vLLM (pin ≥ 0.26, a recipe-verified build). The challenger is
  `Qwen/Qwen3.8-27B-FP8`.
- Settings: `--max-model-len 32768`, `--enable-sleep-mode`, `--reasoning-parser qwen3`, JSON via
  `response_format: json_schema`. Classification, extraction and judge calls run with
  `enable_thinking: false` at temperature 0. Layout proposals and overlay checks use low reasoning
  effort.
- Cheap checks reuse the primary with thinking off, so no extra VRAM is needed.
- Mock: a deterministic schema-valid JSON generator seeded by input hash.
- **Coordinate hints** (ADR-S19) use the model's grounding output (points/boxes) on full-resolution
  tiles with low reasoning effort. They are only search windows for deterministic snapping, never
  geometry. Phase 3 measures hint precision and acceptance rate per model, so the A/B also covers
  grounding quality. (Molmo2-8B, Apache, strong at pointing, is a registered candidate if
  Qwen's grounding underperforms.)
- Rationale: Apache-2.0, not gated, native vision, mature vLLM support. Qwen3.8's image path is not
  yet recipe-verified, so it must win our A/B first.
- Risk: long hidden reasoning with thinking on (a 22k-token case was reported for 3.8). Mitigated by
  thinking off plus `max_tokens` caps.

## ADR-M02: Second-family judge (tie-breaker on critical QA checks)

| Candidate | License (evidence) | Size / VRAM | Runtime | Verdict |
|---|---|---|---|---|
| `google/gemma-4-31B-it` (`google/gemma-4-31B-it-qat-w4a16-ct`; `RedHatAI/gemma-4-31B-it-FP8-dynamic`) | **Apache-2.0**: Google OSS blog: "Gemma 4 models are the first in the Gemmaverse to be released under the OSI-approved Apache 2.0 license" *(snippet)* | QAT W4A16 min 20 GB; FP8 min 38 GB; BF16 75 GB | vLLM ≥ 0.19.1; recipe documents JSON-schema decoding | **Primary judge-2** |
| `mistralai/Ministral-3-14B-Instruct-2512` | Apache-2.0 *(snippet)* | Native FP8, 17 GB | vLLM ≥ 0.11 | **Fallback judge-2** |

**Decision.**
- Gemma 4 31B uses QAT W4A16 on gpu48/gpu80 and FP8-dynamic on gpu96plus. Avoid the llm-compressor
  FP8_BLOCK variant: vLLM issue #39407 (garbage output) was closed without a fix.
- Pin a revision after 2026-07-15, when the checkpoints were refreshed with "a new vision
  configuration".
- It runs in a second vLLM server (`127.0.0.1:8102`), started on demand. Its questions are identical
  binary, schema-constrained questions to the primary's.
- Disagreement on a critical check routes the view to Gate D with both answers shown.

## ADR-M03: OCR and document layout

| Candidate | License (evidence) | Boxes | Languages | Size | Runtime | Quality | Verdict |
|---|---|---|---|---|---|---|---|
| `PaddlePaddle/PaddleOCR-VL-1.6` + `PaddlePaddle/PP-DocLayoutV3_safetensors` | Apache-2.0 (PaddleOCR LICENSE read; card *(snippet)*) | Layout polygons + reading order; `Spotting:` line quads | 109 incl. Arabic, Cyrillic | 0.9B, ~2 GB | vLLM ≥ 0.11.1 **or** transformers backend (PaddleOCR ≥ 3.5); PP-DocLayoutV3 in `transformers` | OmniDocBench v1.6 96.33 (self-reported, PaddleOCR README) | **Primary** (Paddle-free transformers path; if the Paddle pipeline is ever needed, it gets its own venv with cu126 wheels) |
| `zai-org/GLM-OCR` | MIT (README "released under the MIT License") | Region boxes via PP-DocLayoutV3 | zh, en, fr, es, ru, de, ja, ko (no Arabic; Turkish not listed) | 0.9B | vLLM ≥ 0.19 | OmniDocBench v1.5 94.62 | **Fallback** (set `maas.enabled=false`: the SDK defaults to a cloud API) |
| RapidOCR (PP-OCRv5/v6 ONNX) + Docling (MIT) with `docling-project/docling-layout-heron` (Apache-2.0) | Apache/MIT (READMEs, PyPI) | Line-level det+rec boxes; Heron layout boxes | v6: 50 langs; v5 has Arabic rec | Tiny, CPU-capable | ONNX Runtime | v6-medium +5.1% rec over v5-server (README) | **Deterministic fallback + cheap box source** |
| MinerU | "MinerU Open Source License": > 100M MAU / > $20M monthly revenue → separate license; attribution (LICENSE.md read) | — | — | — | — | — | **Blocked** |
| Marker / Surya / Chandra | Weights "AI PUBS OPEN RAIL-M (MODIFIED)": forbidden above $5M revenue/funding (MODEL_LICENSE read) | — | — | — | — | — | **Blocked** |
| HunyuanOCR | "DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA" (LICENSE read) | — | — | — | — | — | **Blocked** |
| Nanonets-OCR2-3B | Derived from Qwen2.5-VL-3B (Qwen Research License) *(snippet)* | — | — | — | — | — | **Blocked** |
| dots.mocr | MIT + "dots.mocr LICENSE AGREEMENT" AUP 3.3(c) (LICENSE read) | JSON bboxes | — | ~3B | vLLM | Vendor Elo | Not selected (AUP needs legal review) |
| olmOCR-2 | Apache-2.0 | **No boxes** | — | 7B | vLLM | — | Unsuitable |

**Decision.**
- Text sources by input type:
  - Vector PDFs: the PDF text layer (exact words + boxes via pypdfium2).
  - DXF: TEXT/MTEXT entities.
  - Raster sheets: PaddleOCR-VL-1.6 on full-resolution tiles + PP-DocLayoutV3 layout. RapidOCR
    line boxes are the deterministic cross-check. Where the two disagree on a dimension string, a
    conflict is raised.
- Office documents go through Docling.
- None of the VLM OCRs yield reliable word boxes. Dimension strings are short line items, so line
  quads suffice, and the tile mapper splits lines into words by glyph gaps when needed.
- Mock: returns the synthetic generator's ground-truth text with boxes.

## ADR-M04: Text-prompted segmentation (QA opening masks, material regions in photos, overlay checks)

| Candidate | License (evidence) | Gated | Size | Runtime | Quality | Verdict |
|---|---|---|---|---|---|---|
| `facebook/sam3` (and `facebook/sam3.1`) | **SAM License** (2025-11-19, LICENSE read). Worldwide, royalty-free, no NC/cap/territory clause. **Conditions:** trade-controls/ITAR/military exclusions; indemnity to Meta; Meta may modify terms "effective immediately"; **no express patent grant** in the sam3 copy | **Yes** (request access on HF) | 848M | transformers `sam3` (5.17) or the official repo | SA-Co/Gold cgF1 54.1 vs OWLv2 24.6; LVIS AP 48.5 (README) | **Primary, `conditional`**: enabled only if the owner accepts `sam-license-2025-11-19` |
| `facebook/sam2.1-hiera-large` + `IDEA-Research/grounding-dino-base` | SAM 2: "checkpoints … licensed under Apache 2.0" (README); Grounding DINO Apache-2.0 (LICENSE) | No | 224M + ~0.2B | transformers `sam2`, `grounding_dino` | — | **Fallback** (fully permissive; Grounding DINO training data mix is a grey area, noted) |
| `microsoft/Florence-2-large` | MIT per HF card *(snippet)* | No | 0.77B | transformers | — | Alternative grounder |
| Ultralytics (YOLO-seg/YOLOE/SAM wrappers), YOLO-World | AGPL-3.0 / GPL-3.0 | — | — | — | — | **Blocked** |
| SegFormer NVIDIA weights; ADE20K/Cityscapes-trained heads | NVIDIA NC license; dataset NC terms | — | — | — | — | **Blocked** |

**Decision.** SAM 3 is primary once its terms are accepted, with a pinned LICENSE snapshot per
download. Until then (and as the fallback), use Grounding DINO + SAM 2.1. QA thresholds are
relative to the base render (ADR-S06), so they remain valid when the fallback is active. Clients
must be screened against sanctions lists before SAM 3 is used on their data.

## ADR-M05: Monocular depth (QA depth-drift metric; concept-mode conditioning)

| Candidate | License (evidence) | Size | Runtime | Quality | Verdict |
|---|---|---|---|---|---|
| `depth-anything/DA3MONO-LARGE`, `depth-anything/DA3METRIC-LARGE` | **Apache-2.0** per-model license column in the Depth-Anything-3 README. `DA3-SMALL` and `DA3-BASE` are also Apache. `DA3-LARGE(-1.1)`, `DA3-GIANT(-1.1)` and `DA3NESTED-*` are **CC BY-NC 4.0** | 0.35B each | `depth-anything-3` (PyPI 0.1.1, Apache) | — | **Primary** |
| `Ruicheng/moge-2-vitl-normal` | MIT (repo LICENSE; card *(snippet)*) | 331M | MoGe package | Metric point map + normals + FOV | **Fallback** |
| `huawei-bayerlab/marigold-v2-0` | Apache-2.0 ("Code and models are released under the Apache License"); backbone Qwen-Image-Edit-2509 (Apache) | 4-bit DiT; ~17 GB at 1024² | Own repo | NYUv2 AbsRel 3.6, ETH3D 2.8 (README); glass see-through depth | Optional quality tier (off by default) |
| Depth Anything V2 Base/Large, UniDepth, Metric3D, Depth Pro, VGGT (orig.) | CC BY-NC / unclear / apple-amlr / NC | — | — | — | **Blocked** |

**Decision.** DA3MONO-LARGE for the relative-depth QA metric (scale/shift aligned), MoGe-2 as
fallback. Only the four Apache DA3 repo ids are allowed. The registry pins those exact ids, and a CI
check rejects any other `depth-anything/*` id.

## ADR-M06: Floor-plan recognition (raster path)

**Finding.** No commercially clean off-the-shelf model exists. Every public checkpoint is trained
on non-commercial or research-only data:
- CubiCasa5K: "Creative Commons Attribution-NonCommercial 4.0" (LICENSE read)
- R2V/LIFULL: academic institutions only
- RPLAN: research-only
- Structured3D: non-commercial research
- FloorPlanCAD: CC BY-NC 4.0 *(snippet)*
- ArchCAD-400K: gated non-commercial *(snippet)*
- FloorplanVLM (arXiv 2602.06507): weights reportedly not released; UNVERIFIED
- Raster2Seq checkpoints: trained on CubiCasa and Structured3D

**Decision.**
1. **v1 baseline (no learned model):** deterministic CV (thick-stroke extraction, skeleton → vector),
   VLM overlay verification, **Gate A mandatory**.
2. **Trained model (Phase 3):** pixel segmentation (wall / door / window / room-interior / symbol /
   background) with a permissive stack.
   - Architecture: `segmentation-models-pytorch` (MIT) U-Net/FPN with a **DINOv2** encoder
     (Apache-2.0 weights) or ConvNeXt (MIT). Symbol detection (door arcs, windows) with RT-DETR
     (Apache-2.0).
   - Training data:
     - Default: our **synthetic plan generator** + the **firm's CAD archive** rasterised with labels
       from DXF layers. Confirm client contracts allow internal ML use (open question Q-5).
     - Optional `conditional` data: ResPlan (CC BY 4.0 data, MIT code; scraped-listing provenance)
       and Modified Swiss Dwellings (CC BY 4.0 *(snippet)*). Used only if the owner accepts.
   - The training script runs on the pod. The weights are the firm's own and are registered as
     `license: proprietary-firm`, `commercial_ok: true`.
3. **Forbidden:** any checkpoint trained on the datasets above; Ultralytics; YOLO-World; SegFormer
   NVIDIA weights.

Promotion rule: the trained checkpoint replaces the CV baseline only when it beats it on the
held-out synthetic + firm set, and only after it meets the clean-raster targets (wall F1 ≥ 0.92,
opening F1 ≥ 0.88). Only then may Gate A become policy-driven for raster plans.

## ADR-M07: Image/text embeddings (material & style retrieval, mood-board similarity)

| Candidate | License (evidence) | Size | Verdict |
|---|---|---|---|
| `google/siglip2-so400m-patch16-384` | Apache-2.0 (card *(snippet)*; big_vision LICENSE read) | ~1.1B total, ~2–4 GB | **Primary** |
| `facebook/PE-Core-L14-336` | Apache-2.0 ("released with the Apache 2.0 license", perception_models README). Do not confuse with the PLM research-licensed models in the same repo | L/14 | **Fallback** |
| `Qwen/Qwen3-VL-Embedding-2B` | Apache-2.0 (LICENSE read) | 2B, ~5 GB | Optional (document-image retrieval, reranker) |
| DINOv3 | DINOv3 License: trade-controls clause; gated with manual review | — | Not selected (legal review) |
| jina-clip-v2, MetaCLIP 2, DFN5B, LAION OpenCLIP | CC BY-NC / apple-amlr / "deployed use … out of scope" | — | **Blocked / avoided** |

## ADR-M08: Faithful refinement (S8, client default)

**Verified facts (diffusers source a12c38a + wheels):**
- `QwenImageEditPlusPipeline` (since 0.36.0) supports `Qwen/Qwen-Image-Edit-2511` from
  **diffusers 0.37.0** (PR #12839 `zero_cond_t`). Pin: **0.40.0**.
- It has **no `strength` and no `mask` argument**: it denoises from pure noise, with the references
  as context tokens.
- "Native ControlNet since 2509" means **in-context conditioning**: a depth or edge map is passed as
  an extra image in `image=[…]` and referenced in the prompt. There is no ControlNet module and no
  pixel-alignment guarantee.
- The docs say 1–3 input images work best. Training covers ~1–1.8 MP, so 4K needs tiling.

| Candidate | License (evidence) | VRAM / speed evidence | Quality evidence | Verdict |
|---|---|---|---|---|
| `Qwen/Qwen-Image-Edit-2511` (20B MMDiT + Qwen2.5-VL-7B encoder + Qwen VAE) | **Apache-2.0** (Qwen-Image README "licensed under Apache 2.0"; LICENSE read; card *(snippet)*). Released 2025-12-23 | BF16 transformer ~40.9 GB. Measured peak **61,091 MiB** at 1024², 50 steps on 1×H200 for the same 20B stack (vLLM-Omni recipe). ~0.34 s/step without CFG; ~2× with true-CFG | GEdit-EN 7.877, ImgEdit 4.51 (third-party table in the FireRed README) | **Primary** |
| `lightx2v/Qwen-Image-Edit-2511-Lightning` (4/8-step LoRA) | Apache-2.0 (card *(snippet)*; ModelTC LICENSE read) | ~850 MB | Distilled; FP8 "grid" artifacts if the base is naively downcast | **Previews / Gate C only.** Finals use full steps unless an A/B passes QA |
| `FireRedTeam/FireRed-Image-Edit-1.0/1.1` | Apache-2.0 ("code and the weights … Apache 2.0", LICENSE read) | Same pipeline class | Self-reported GEdit-EN 7.943 | A/B challenger |
| `Qwen/Qwen-Image-2512` + `InstantX/Qwen-Image-ControlNet-Union` (canny, soft edge, depth, pose) | Apache-2.0 / Apache-2.0 (card *(snippet)*) | Qwen-Image-2512: 58,979 MiB peak, 7.7 s for 20 steps at 1024² on A800 | — | **Fallback** (strongest structural lock; needs our img2img+ControlNet pipeline) |
| `black-forest-labs/FLUX.2-klein-4B` | **Apache-2.0** (BFL flux2 README license table, read) | "fits in ~8GB" (BFL); ~13 GB third-party | — | **Fast fallback.** `Flux2KleinInpaintPipeline` (0.38.0) natively takes `image`, `image_reference`, `mask_image`, `strength` |

**Decision: our pipeline code over the official classes (ADR-S07)**
- **Strength:** encode the Cycles render to latents, noise to σ₀ (strength), and run a truncated
  sigma schedule.
- **Per-pixel strength map:** each step, blend in `callback_on_step_end` against the render latents
  noised to the current σ (differential-diffusion style; no Qwen community pipeline exists, so this
  is our port).
- **Inputs:** the render tile + a depth/edge map (in-context, "Picture 2") + at most one mood
  reference.
- **Structural lock escalation:** if QA fails on geometry after the strength-reduction retries, a
  **hard composite** follows as one more retry rung. The Cycles pixels inside structural masks are
  kept, with gradient-domain blending at the mask edges, before the final fallback to the pure
  Cycles render.
- **Tiling:** our MultiDiffusion-style latent tiler (≈1328 px tiles, 256 px overlap, one global noise
  field, cosine weights), then LAB colour matching back to the base.
- **Settings:** finals use 30–40 steps, `true_cfg_scale` 4. Previews use Lightning 8-step.
- **Mock:** a deterministic unsharp + colour-grade op that keeps geometry exactly.

## ADR-M09: Concept mode generation

**Decision.**
- Primary: `Qwen/Qwen-Image-2512` + `InstantX/Qwen-Image-ControlNet-Union` via
  `QwenImageControlNetPipeline` (depth + canny from the S7 passes), rendered at 1664×928, then our
  tiler.
- Fallback: Qwen-Image-Edit-2511 with multiple references.
- Fast fallback: `Tongyi-MAI/Z-Image-Turbo` (Apache-2.0, LICENSE read) +
  `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.x` (Apache *(snippet)*) via
  `ZImageControlNetInpaintPipeline` (0.37.0).
- Every output is stamped "concept — not dimensionally verified".

## ADR-M10: Disabled image models (recorded so nobody re-enables them by accident)

| Model | Reason (evidence) |
|---|---|
| `Qwen/Qwen-Image-2.1` (2026-09-20) | **Qwen RESEARCH LICENSE**: "FOR NON-COMMERCIAL PURPOSES ONLY" (LICENSE read) |
| Qwen-Image-2.0, Qwen-Image-3.0 / 3.0 Pro | API-only, no weights |
| FLUX.2 [dev], [klein] 9B / 9B-KV / base-9B; FLUX.1 Kontext [dev] | FLUX Non-Commercial License: "revenue-generating activity … is not a Non-Commercial Purpose" (LICENSE read). Enabled only with a configured BFL commercial license |
| FLUX.2 [pro]/[flex]/[max] | API-only |
| HunyuanImage 2.1/3.0, Hunyuan3D, HunyuanOCR | Tencent license excludes EU/UK/South Korea + MAU clause (LICENSE read) |
| HiDream-I1/E1 | Text encoder Llama-3.1-8B (700M-MAU clause) |
| SD 3.5 | Stability Community License ($1M revenue cap) |
| OmniGen2 | Depends on Qwen2.5-VL-3B (Qwen Research License) |
| Ideogram 4; Krea 2; Sana | Non-commercial / revenue+seat caps / NVIDIA license |
| Step1X-Edit v1.2; Bagel | Forked or no diffusers runtime |
| Nunchaku (diffusers "Nunchaku Lite") | Not supported on Hopper; kernels from an untrusted publisher; 2511/2512 weights third-party only |

## ADR-M11: Upscaler

| Candidate | License (evidence) | Verdict |
|---|---|---|
| HAT `Real_HAT_GAN_SRx4` (XPixelGroup/HAT) | Apache-2.0 (LICENSE read; OpenModelDB) | **Primary.** Deterministic, geometry-faithful. Used to lift the ~1.3 MP global refine to 4K before the tiled low-strength pass guided by the native 4K Cycles render |
| Real-ESRGAN `RealESRGAN_x2plus` / `x4plus` | BSD-3-Clause (LICENSE read) | **Fallback** |
| `ByteDance-Seed/SeedVR2-3B` | Apache-2.0 (README) | Off. It hallucinates detail ("overly generate details" on light degradations, README), which conflicts with faithful mode. Concept mode only, after an A/B |
| 4x-UltraSharp, SUPIR, HYPIR, StableSR, AuraSR (CC BY-SA) | NC / proprietary / S-Lab / ShareAlike | **Blocked** |

## ADR-M12: Aesthetic / IQA ranker (ranks candidates only; never gates)

**Finding.** **pyiqa is PolyForm Noncommercial 1.0.0** (repo LICENSE; PyPI 0.1.16
`license_expression`), and its older PyPI "Apache" classifiers contradict the repo. **pyiqa is
therefore blocked**, and with it TOPIQ, CLIP-IQA, and the MUSIQ/MANIQA ports.

**Decision.** Ranking score = LAION improved-aesthetic-predictor (Apache-2.0 LICENSE read; on OpenAI
CLIP ViT-L/14, MIT) + a **firm-trained preference head** (small MLP on SigLIP 2 embeddings) trained
on Gate D accept/reject history once ≥ 200 decisions exist. Optional: RALI / `ByteDance/Q-Insight`
(Apache). Blocked: pyiqa, Q-Align, DeQA-Score (Llama-2 base), aesthetic-predictor-v2.5 (AGPL).
Technical IQA (noise, banding, sharpness, clipping) is our own deterministic code, not a model.

## ADR-M13: Image-to-3D (optional, client furniture photos)

**Finding.**
- `microsoft/TRELLIS.2-4B` is MIT, but its GLB export imports **nvdiffrast**, which is under the
  NVIDIA Source Code License: "non-commercially … research or evaluation purposes only".
- Its pipeline references `briaai/RMBG-2.0` (CC BY-NC).
- SAM 3D Objects' mesh path also defaults to nvdiffrast, and its requirements include smplx (MPI NC).
- TripoSG bundles Tencent-licensed code. SF3D/SPAR3D are under the Stability $1M cap. Hunyuan3D is
  excluded.

**Decision.** **Disabled in v1.** The `ImageTo3D` role exists with a mock. The registry records
TRELLIS.2 as `blocked` until: (1) background removal is swapped to BiRefNet (MIT), (2) GLB texture
baking is done without nvdiffrast (Blender bake or PyTorch3D, BSD), and (3) the DINOv3 terms are
accepted. Geometry-only fallback candidate: `wushuang98/Direct3D-S2` (MIT). Generated assets are
always flagged `generated` and shown at Gate B/C.

## ADR-M14: Deterministic components that replace models (not model ADRs, recorded for completeness)
- Edge detection for QA uses multi-scale Canny (OpenCV, Apache-2.0), not a learned edge model (HED
  and PiDiNet weights have unclear data terms).
- Line segments use OpenCV LSD (the Apache-2.0 re-implementation in OpenCV ≥ 4.5.1).
- Sun position uses pvlib (see Tools).
- Palette extraction is k-means in CIELAB (scikit-learn, BSD).

---

## Tools, libraries and assets (license audit baseline)
Versions are PyPI latest on 2026-09-30. "OK" means acceptable in an internal, non-distributed
commercial network service. `scripts/license_audit.py` enforces this list per environment (app,
vLLM, Blender, UI npm) and generates `THIRD_PARTY_LICENSES.md`.

| Component | Version | License | OK? | Notes |
|---|---|---|---|---|
| Blender (subprocess) | 5.2.2 LTS | GPL-3.0-or-later binaries (Cycles Apache-2.0) | ✅ | `archrender_blender/` is GPL-3.0-or-later with SPDX headers and imports no app modules. Obligations arise only on distribution. |
| LibreDWG `dwg2dxf` (subprocess) | 0.14 | GPL-3.0-or-later | ✅ | Arm's-length CLI; source offer needed only if the image is distributed |
| ODA File Converter | — | ODA terms: non-members "non-commercial applications only" *(snippet)* | ❌ | Unless the firm is an ODA member |
| ifcopenshell | 0.9.0 | LGPL-3.0-or-later | ✅ | Unmodified library use |
| ezdxf | 1.4.4 | MIT | ✅ | |
| pypdfium2 (+ PDFium) | 5.13.0 | Apache-2.0 / BSD-3 | ✅ | **Not thread-safe**, so it runs in the process pool |
| pdfplumber / pdfminer.six | 0.11.10 / 20260107 | MIT | ✅ | |
| PyMuPDF | 1.28.2 | AGPL-3.0 / commercial | ❌ | Unless an Artifex licence is configured |
| pdf2image (→ poppler-utils GPL), OCRmyPDF (→ Ghostscript AGPL) | — | MIT/MPL wrappers of GPL/AGPL tools | ❌ | Not used; pypdfium2 renders pages |
| manifold3d / trimesh / shapely | 3.5.4 / 5.1.0 / 2.1.2 | Apache-2.0 / MIT / BSD-3 (GEOS LGPL, dynamic) | ✅ | |
| pvlib | 0.16.1 | BSD-3 | ✅ | NREL SPA; `apparent_elevation` |
| Pillow | 12.3.0 | MIT-CMU | ✅ | |
| pillow-heif | 1.8.0 | BSD-3 source, but wheels bundle **x265 (GPL-2)** | ⛔ not used | HEIC via libheif `heif-dec` with the decoder plugin only (ADR-S17) |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (bundled FFmpeg LGPL-2.1) | ✅ | |
| numpy / scipy / scikit-image / scikit-learn / colour-science | current | BSD-3 | ✅ | |
| FastAPI / uvicorn / pydantic / sse-starlette | 0.142.2 / 0.54.0 / 2.13.5 / 3.5.0 | MIT / BSD-3 / MIT / BSD-3 | ✅ | |
| SQLAlchemy / aiosqlite | 2.1.1 / 0.22.1 | MIT | ✅ | |
| Litestream | ≥ 0.5.2 | Apache-2.0 | ✅ | |
| Redis server ≥ 8 | — | RSALv2 / SSPLv1 / AGPLv3 | ❌ | Not used (Valkey BSD-3 if ever needed) |
| supervisor | 4.3.0 | BSD-derived (repoze) | ✅ | |
| docling (+ core/parse/ibm-models) | 2.131.0 | MIT | ✅ | Its model weights are listed in the model registry |
| RapidOCR / onnxruntime | current | Apache-2.0 / MIT | ✅ | |
| python-docx / openpyxl / python-pptx | 1.2.0 / 3.1.5 / 1.0.2 | MIT | ✅ | Macros are never executed |
| WeasyPrint / Jinja2 | 70.0 / current | BSD-3 | ✅ | System Pango (LGPL) |
| torch | 2.14.0 (app), 2.13.0 (vLLM) | BSD-style; NVIDIA CUDA wheels under the NVIDIA EULA (redistributable) | ✅ | |
| diffusers / transformers / accelerate / huggingface_hub | 0.40.0 / 5.17.0 / 1.15.0 / 2.0.0 | Apache-2.0 | ✅ | diffusers 0.40 requires `huggingface-hub<2.0`, so the hub is pinned to 1.x in the app venv |
| vLLM / xgrammar | 0.30.0 / 0.2.8 | Apache-2.0 | ✅ | |
| torchao / flash-attn / xformers | 0.18.0 / 2.8.3.post1 / 0.0.35 | BSD-3 | ✅ | |
| **pyiqa** | 0.1.16 | **PolyForm-Noncommercial-1.0.0** | ❌ | Blocked (ADR-M12) |
| lpips | 0.1.4 | BSD-2 | ✅ (offline eval only) | ImageNet backbone weights; never used as a gate |
| nvdiffrast / nvdiffrec | — | NVIDIA Source Code License (non-commercial) | ❌ | Blocks TRELLIS.2 / SAM 3D mesh export (ADR-M13) |
| smplx | — | MPI non-commercial | ❌ | |
| Ultralytics | — | AGPL-3.0 | ❌ | |
| rhino3dm | 8.35.0 | MIT | ✅ | `.3dm` ingestion |
| svgelements | 1.9.6 | MIT | ✅ | SVG parsing; cairosvg (LGPL-3) is avoided |
| puremagic / filetype | 2.2.0 / 1.2.0 | MIT | ✅ | Magic-byte detection |
| defusedxml | current | PSF-2.0 | ✅ | |
| three.js / React / Vite | 0.186.1 / 19.3.0 / 8.3.1 | MIT | ✅ | |
| Hypothesis / pytest / ruff / mypy / pip-licenses | dev | MPL-2.0 / MIT / MIT / MIT / MIT | ✅ | Dev-only |
| Poly Haven assets | — | CC0 (API ToS: credit "Powered by Poly Haven" + identifying User-Agent) | ✅ | Mirrored once with provenance |
| ambientCG assets | — | CC0 1.0 *(snippet)* | ✅ | API v2; mirrored once |
| RAL/NCS colour tables | — | Varies | ⚠️ | Only openly licensed conversion data; results flagged as approximations |

**Distribution note.** If the Docker image is ever shipped outside the firm, it contains GPL
components (Blender, LibreDWG, and libheif if built with GPL plugins). Written source offers are then
required. The current plan is internal use only.

## Registry sketch (`configs/models.yaml`, validated by pydantic; pins in `models.lock.yaml`)
```yaml
- role: refiner
  name: qwen-image-edit-2511
  repo: Qwen/Qwen-Image-Edit-2511
  revision: null            # pinned by scripts/pin_models.py → models.lock.yaml
  license: {id: apache-2.0, class: permissive,
            evidence_url: https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/LICENSE,
            checked_at: 2026-09-30}
  territorial_exclusions: []
  caps: {}
  gated: false
  runtime: diffusers>=0.40.0,<0.41
  pipeline: archrender.refine.qwen_edit_plus:FaithfulQwenEditPlus
  vram_gb: {bf16: 61, fp8: null}       # fp8 only after A/B (Phase 6)
  disk_gb: 58
  fallback: qwen-image-2512-controlnet-union
  profiles: [gpu48, gpu80, gpu96plus]
```

## VRAM and disk per profile
See [PLAN.md § Budgets](PLAN.md#budgets-per-hardware-profile). The profile files are the executable
form of that table.
