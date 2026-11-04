# Meta API notes

## Documentation-confirmed

- The Meta API base URL is `https://api.meta.ai/v1`; Responses authentication uses the model API key.
- `POST /v1/responses/input_tokens` reports `input_tokens` for a supported Responses request and includes multimodal/system context.
- Files uploads use multipart `file` with `purpose=user_data`; a SAM video request references the returned file ID as `input_video.file_id`.
- The SAM request model is `sam-3.1`. Its streamed output is parsed with `meta-sam-parser` public APIs (`video_segmentation_format`, parser `push`, and `finish(CompletedOutcome())`), rather than token regular expressions.
- Documented Standard pricing configured for the Phase A guard: Wonder input/output/cached-input is $4/$20/$0.20 per million tokens; Spark is $1.25/$4.25/$0.15 per million; SAM is $0.0025/image and $0.0002/video frame with a 50-frame minimum.

Primary sources: [API reference](https://dev.meta.ai/docs/api-reference), [Responses](https://dev.meta.ai/docs/protocols/responses), [structured output](https://dev.meta.ai/docs/structured-output), [token counting](https://dev.meta.ai/docs/token-counting), [Files](https://dev.meta.ai/docs/file-handling), [models](https://dev.meta.ai/docs/models), [pricing](https://dev.meta.ai/docs/pricing-rate-limits), [SAM segmentation](https://dev.meta.ai/docs/sam/segmenting), [official parser](https://github.com/meta-models/meta-sam/blob/main/python/README.md).

## Live-confirmed

`scripts/spike_api.py` is opt-in. Successful synthetic confirmations are recorded below; a missing key exits blocked without dispatch or spend.

## Unverified

The synthetic spike below confirms basic API/schema/image/upload/stream/parser compatibility and observed billing. Files-ID lifetime, real-video quality, box prompting, long-video stability and live rate-limit recovery remain unverified. Documentation evidence is not live evidence for those paths.

## Implemented request boundaries

- Responses JSON schemas use `text.format`, not `response_format`; every object is closed and all nullable properties are explicit. Planner patches enumerate allowed parameter names.
- Images use `input_image.image_url` with authorized image bytes. Standard models are used, not Contributor variants.
- SAM noun phrases and box grammar are separate calls. Box builders consume half-open bounds and emit inclusive wire endpoints. The official documented cropped-mask fixture was parsed incrementally and placed in source coordinates in tests; that is not a live SAM call.
- SDK retries are disabled; bounded tenacity retries are charged/logged individually. Failed or unmetered attempts keep conservative reservations. Cached-input discounts are not assumed in the ledger estimate.
- The spike writes the exact successful synthetic wire request structures, response fixtures, usage and ledger into `runs/spike-api/observed_fixture.json` only after observed success. No credentials are included. B1 refuses real-video dispatch until that evidence exists.
- All production callers share `runs/meta_budget.json`, even if run output directories change. Configuration can lower but cannot raise the $10 Phase A / $50 development hard caps or lower the $70 reserve.

## Optional SAM 3D contract

The README's `output["gs"].save_ply()` example is not the only output. The official [pipeline mesh postprocessing](https://github.com/facebookresearch/sam-3d-objects/blob/main/sam3d_objects/pipeline/inference_pipeline.py) constructs `outputs["glb"]`, which the [pointmap pipeline](https://github.com/facebookresearch/sam-3d-objects/blob/main/sam3d_objects/pipeline/inference_pipeline_pointmap.py) returns. The optional server exports that real mesh through trimesh. No weights or GPU inference were loaded in Phase A; only the explicitly labeled CPU stub and HTTP/cache integration were exercised. Real model runtime, gated weights/license and provider lifecycle remain Phase B prerequisites.


## Live-confirmed spike

- UTC: `2026-11-04T16:26:55+00:00`
- Observed successful Wonder JSON-schema, Spark image, Files upload, and streamed SAM requests. Local fixture: `runs/spike-api/observed_fixture.json`; committed copy: `tests/fixtures/meta_api_spike.json`. Replaying the observed SAM text through the official parser yields two boxes and two masks.
- Ledger snapshot recorded at completion: `{"committed_usd": "0.028378", "dispatch_blocked": false, "pending_reservations": 0, "pending_usd": "0", "phase_totals": {"A": {"committed_usd": "0.028378", "pending_usd": "0", "total_usd": "0.028378"}}, "total_usd": "0.028378"}`. This note records format evidence only; it is not a quality or real-video claim.

Working wire requests (synthetic media only, bearer header omitted):

```json
{
  "files": {
    "file_artifact": "runs/spike-api/synthetic_media/synthetic-spike.mp4",
    "file_sha256": "a968d71092d7b56799f4ce64212bd710d57853c5b01990bc2439de459de5aa23",
    "method": "POST",
    "purpose": "user_data",
    "url": "https://api.meta.ai/v1/files"
  },
  "sam": {
    "input": [
      {
        "content": [
          {
            "text": "colored rectangle",
            "type": "input_text"
          },
          {
            "file_id": "file-1455888273181349",
            "type": "input_video"
          }
        ],
        "role": "user",
        "type": "message"
      }
    ],
    "model": "sam-3.1",
    "stream": true
  },
  "spark": {
    "input": [
      {
        "content": [
          {
            "text": "synthetic image only",
            "type": "input_text"
          },
          {
            "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAuklEQVRIDbXBQQEAEBQFwX1ZRJBOAulEkGBLCMDF4c9EpVJUKkWlUlQqRaVSVCpFpVJUKkWlUlQqReVlj86nNheXqLzs0fnU5uISlZc9Op/aXFyi8rJH51Obi0tUXvbofGpzcYnKyx6dT20uLlF52aPzqc3FJSove3Q+tbm4ROVlj86nNheXqLzs0fnU5uISlZc9Op/aXFyi8rJH51Obi0tUKkWlUlQqRaVSVCpFpVJUKkWlUlQqRaXSAS6TfEEnbJ5AAAAAAElFTkSuQmCC",
            "type": "input_image"
          }
        ],
        "role": "user"
      }
    ],
    "instructions": "List the single synthetic colored rectangle as an object.",
    "max_output_tokens": 2048,
    "model": "muse-spark-1.3",
    "text": {
      "format": {
        "name": "objectlist",
        "schema": {
          "$defs": {
            "ObjectInfo": {
              "additionalProperties": false,
              "properties": {
                "approx_size_cm": {
                  "items": {
                    "type": "number"
                  },
                  "maxItems": 3,
                  "minItems": 3,
                  "title": "Approx Size Cm",
                  "type": "array"
                },
                "count": {
                  "maximum": 20,
                  "minimum": 1,
                  "title": "Count",
                  "type": "integer"
                },
                "graspable": {
                  "title": "Graspable",
                  "type": "boolean"
                },
                "id": {
                  "title": "Id",
                  "type": "string"
                },
                "material_guess": {
                  "title": "Material Guess",
                  "type": "string"
                },
                "name": {
                  "title": "Name",
                  "type": "string"
                },
                "schema_version": {
                  "title": "Schema Version",
                  "type": "integer"
                }
              },
              "required": [
                "schema_version",
                "id",
                "name",
                "material_guess",
                "approx_size_cm",
                "graspable",
                "count"
              ],
              "title": "ObjectInfo",
              "type": "object"
            }
          },
          "additionalProperties": false,
          "properties": {
            "objects": {
              "items": {
                "$ref": "#/$defs/ObjectInfo"
              },
              "title": "Objects",
              "type": "array"
            },
            "schema_version": {
              "title": "Schema Version",
              "type": "integer"
            },
            "source": {
              "enum": [
                "synthetic",
                "recorded",
                "live",
                "fallback"
              ],
              "title": "Source",
              "type": "string"
            }
          },
          "required": [
            "schema_version",
            "objects",
            "source"
          ],
          "title": "ObjectList",
          "type": "object"
        },
        "strict": true,
        "type": "json_schema"
      }
    }
  },
  "wonder": {
    "input": "Synthetic tabletop format check only.",
    "instructions": "Return a bounded scenario family for a synthetic tabletop scene.",
    "max_output_tokens": 2048,
    "model": "muse-wonder-1.0",
    "text": {
      "format": {
        "name": "scenariofamily",
        "schema": {
          "additionalProperties": false,
          "properties": {
            "clutter_count": {
              "items": {
                "type": "integer"
              },
              "maxItems": 2,
              "minItems": 2,
              "title": "Clutter Count",
              "type": "array"
            },
            "friction_scale": {
              "items": {
                "type": "number"
              },
              "maxItems": 2,
              "minItems": 2,
              "title": "Friction Scale",
              "type": "array"
            },
            "jitter_m": {
              "maximum": 0.04,
              "minimum": 0,
              "title": "Jitter M",
              "type": "number"
            },
            "mass_scale": {
              "items": {
                "type": "number"
              },
              "maxItems": 2,
              "minItems": 2,
              "title": "Mass Scale",
              "type": "array"
            },
            "occluder_count": {
              "items": {
                "type": "integer"
              },
              "maxItems": 2,
              "minItems": 2,
              "title": "Occluder Count",
              "type": "array"
            },
            "occluder_height": {
              "items": {
                "type": "number"
              },
              "maxItems": 2,
              "minItems": 2,
              "title": "Occluder Height",
              "type": "array"
            },
            "schema_version": {
              "title": "Schema Version",
              "type": "integer"
            },
            "yaw_range": {
              "items": {
                "type": "number"
              },
              "maxItems": 2,
              "minItems": 2,
              "title": "Yaw Range",
              "type": "array"
            }
          },
          "required": [
            "schema_version",
            "jitter_m",
            "yaw_range",
            "friction_scale",
            "mass_scale",
            "clutter_count",
            "occluder_count",
            "occluder_height"
          ],
          "title": "ScenarioFamily",
          "type": "object"
        },
        "strict": true,
        "type": "json_schema"
      }
    }
  }
}
```

## Live B1 box-prompt confirmation

After the laptop noun-phrase path failed, a separate SAM box request worked on the authorized 480×864 phone video. The official `build_object_prompt` received `size=(480,864)`, one object with ID 1, frame 0 and half-open box `(214,300,424,582)`. Its emitted token text was sent as the sole `input_text` alongside the uploaded `input_video.file_id`, never combined with the noun phrase. The completed stream parsed into 336 box and 336 mask records. These are real model predictions from an operator-provided box, not synthetic masks. Raw real-video payloads and credentials are not published.

