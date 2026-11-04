# Meta API notes

## Documentation-confirmed

- The Meta API base URL is `https://api.meta.ai/v1`; Responses authentication uses the model API key.
- `POST /v1/responses/input_tokens` reports `input_tokens` for a supported Responses request and includes multimodal/system context.
- Files uploads use multipart `file` with `purpose=user_data`; a SAM video request references the returned file ID as `input_video.file_id`.
- The SAM request model is `sam-3.1`. Its streamed output is parsed with `meta-sam-parser` public APIs (`video_segmentation_format`, parser `push`, and `finish(CompletedOutcome())`), rather than token regular expressions.
- Documented Standard pricing configured for the Phase A guard: Wonder input/output/cached-input is $4/$20/$0.20 per million tokens; Spark is $1.25/$4.25/$0.15 per million; SAM is $0.0025/image and $0.0002/video frame with a 50-frame minimum.

Primary sources: [API reference](https://dev.meta.ai/docs/api-reference), [Responses](https://dev.meta.ai/docs/protocols/responses), [structured output](https://dev.meta.ai/docs/structured-output), [token counting](https://dev.meta.ai/docs/token-counting), [Files](https://dev.meta.ai/docs/file-handling), [models](https://dev.meta.ai/docs/models), [pricing](https://dev.meta.ai/docs/pricing-rate-limits), [SAM segmentation](https://dev.meta.ai/docs/sam/segmenting), [official parser](https://github.com/meta-models/meta-sam/blob/main/python/README.md).

## Live-confirmed

No live calls have been made in this checkout. `scripts/spike_api.py` is opt-in and records an appended entry here only after all three synthetic requests and the SAM upload/stream complete with observed responses. A missing key exits blocked without dispatch or spend.

## Unverified

Until a successful opt-in spike is recorded, API compatibility, actual token/frame usage, Files-ID lifetime, Spark image behavior, and streamed SAM parser output remain unverified. Documentation evidence is not live evidence.

## Implemented request boundaries

- Responses JSON schemas use `text.format`, not `response_format`; every object is closed and all nullable properties are explicit. Planner patches enumerate allowed parameter names.
- Images use `input_image.image_url` with authorized image bytes. Standard models are used, not Contributor variants.
- SAM noun phrases and box grammar are separate calls. Box builders consume half-open bounds and emit inclusive wire endpoints. The official documented cropped-mask fixture was parsed incrementally and placed in source coordinates in tests; that is not a live SAM call.
- SDK retries are disabled; bounded tenacity retries are charged/logged individually. Failed or unmetered attempts keep conservative reservations. Cached-input discounts are not assumed in the ledger estimate.
- The spike writes the exact successful synthetic wire request structures, response fixtures, usage and ledger into `runs/spike-api/observed_fixture.json` only after observed success. No credentials are included. B1 refuses real-video dispatch until that evidence exists.
- All production callers share `runs/meta_budget.json`, even if run output directories change. Configuration can lower but cannot raise the $10 Phase A / $50 development hard caps or lower the $70 reserve.

## Optional SAM 3D contract

The README's `output["gs"].save_ply()` example is not the only output. The official [pipeline mesh postprocessing](https://github.com/facebookresearch/sam-3d-objects/blob/main/sam3d_objects/pipeline/inference_pipeline.py) constructs `outputs["glb"]`, which the [pointmap pipeline](https://github.com/facebookresearch/sam-3d-objects/blob/main/sam3d_objects/pipeline/inference_pipeline_pointmap.py) returns. The optional server exports that real mesh through trimesh. No weights or GPU inference were loaded in Phase A; only the explicitly labeled CPU stub and HTTP/cache integration were exercised. Real model runtime, gated weights/license and provider lifecycle remain Phase B prerequisites.

