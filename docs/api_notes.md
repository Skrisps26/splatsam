# Meta API notes

## Documentation-confirmed

- The Meta API base URL is `https://api.meta.ai/v1`; Responses authentication uses the model API key.
- `POST /v1/responses/input_tokens` reports `input_tokens` for a supported Responses request and includes multimodal/system context.
- Files uploads use multipart `file` with `purpose=user_data`; a SAM video request references the returned file ID as `input_video.file_id`.
- The SAM request model is `sam-3.1`. Its streamed output is parsed with `meta-sam-parser` public APIs (`video_segmentation_format`, parser `push`, and `finish(CompletedOutcome())`), rather than token regular expressions.
- Documented Standard pricing configured for the Phase A guard: Wonder input/output/cached-input is $4/$20/$0.20 per million tokens; Spark is $1.25/$4.25/$0.15 per million; SAM is $0.0025/image and $0.0002/video frame with a 50-frame minimum.

Sources recorded in `docs/decisions.md`: Meta API reference, token-counting, pricing/rate-limit, Files, and SAM segmenting documentation.

## Live-confirmed

No live calls have been made in this checkout. `scripts/spike_api.py` is opt-in and records an appended entry here only after all three synthetic requests and the SAM upload/stream complete with observed responses. A missing key exits blocked without dispatch or spend.

## Unverified

Until a successful opt-in spike is recorded, API compatibility, actual token/frame usage, Files-ID lifetime, Spark image behavior, and streamed SAM parser output remain unverified. Documentation evidence is not live evidence.
