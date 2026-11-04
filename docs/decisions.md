# Library-first decisions

Searches performed before implementation; no inference or paid resources during research.

| Component | Sources searched | Choice and reason |
| --- | --- | --- |
| Physics/renderer | https://mujoco.readthedocs.io/en/stable/python.html ; https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rendering/classic/renderer.py | official mujoco CPU, dynamic free hand welded to mocap; segmentation is (geom ID, type), not body ID |
| Numerical statistics | https://numpy.org/doc/stable/ ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html ; https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html | numpy seeded RNG, paired bootstrap, statsmodels Wilson; no handwritten interval formula |
| Geometry/planning | https://trimesh.org/trimesh.bounds.html ; https://github.com/ShirakuGIT/grasp-sampler ; https://github.com/NVlabs/DefGraspSim/tree/main/graspsampling-py-defgraspsim | trimesh OBB/transforms; no verified drop-in native-primitive floating-hand planner, use small principal-axis glue |
| Convex decomposition | https://github.com/SarahWeiii/CoACD | optional coacd; trimesh convex hull when wheel/decomposition unavailable; no mesh needed for primitives |
| Mask processing | https://docs.opencv.org/4.x/ | opencv-contrib-python, homography/contours/min-area rectangles; one OpenCV distribution |
| Meta | https://github.com/openai/openai-python ; https://dev.meta.ai/docs/api-reference ; https://dev.meta.ai/docs/sam/segmenting | official OpenAI SDK with explicit Meta URL/key, no invented API |
| SAM parser | https://pypi.org/project/meta-sam-parser/ ; https://github.com/meta-models/meta-sam/blob/main/python/README.md | meta-sam-parser 0.1.0 public token parser/builders, not custom regex |
| Schemas/cache/retry | https://docs.pydantic.dev/latest/ ; https://grantjenks.com/docs/diskcache/ ; https://tenacity.readthedocs.io/ | Pydantic v2 boundaries, diskcache hashes, bounded tenacity network retries |
| Tests/tools | https://docs.pytest.org/ ; https://docs.astral.sh/ruff/ | pytest behavioral tests, ruff developer checks |
| Dashboard/video | https://docs.streamlit.io/ ; https://plotly.com/python-api-reference/generated/plotly.express.imshow.html ; https://imageio.readthedocs.io/ ; https://github.com/imageio/imageio-ffmpeg | Streamlit read-only viewer, Plotly; imageio/ffmpeg genuine MuJoCo media |
| Remote service | https://fastapi.tiangolo.com/tutorial/request-files/ ; https://www.python-httpx.org/ | FastAPI/uvicorn multipart HTTP, httpx, CPU stub with identical wire contract |
| SAM 3D wrapper | https://github.com/facebookresearch/sam-3d-objects ; https://github.com/tas1337/sam-3d-objects-api-wrapped | no trusted official prebuilt HTTP image found; community image unverified, avoid it; optional reviewed upstream backend only |
| GPU lifecycle | https://docs.runpod.io/sdks/graphql/manage-pods | RunPod user-selected but MCP not mounted; no Phase A provisioning; optional Phase B needs approval and verified termination |
| Segmentation outage | https://github.com/facebookresearch/sam2 | optional SAM 2.1 tiny CPU predictor only if prolonged live SAM outage, not a default weight download |

Python 3.12 selected for scientific-wheel compatibility. Installed exact versions are recorded in uv.lock. Defaults use Standard Meta models rather than Contributor variants. No MODEL_API_KEY was present at project creation; the later user-provided ignored `.env` enabled the bounded live synthetic spike. Public GitHub requested; raw video/private data and secrets remain excluded. Scenario physics multipliers are hidden from planner's nominal mass/friction estimates.

## Implementation evidence and corrections

- Software EGL/llvmpipe renders genuine MuJoCo RGB and segmentation locally; OSMesa was unavailable. Renderer initialization precedes MuJoCo imports. No inference GPU or paid compute was used.
- The calibration fixture uses a distant approximately 3 m camera, 23° field of view and 1280×960 frames. This reduces height parallax and narrow-marker quantization without changing the required 5% scale / 10% size acceptance thresholds.
- Digital contour perimeter biased circularity. OpenCV contour simplification at 0.75 raster pixels restores a sphere classification while retaining the requested 0.85 circularity threshold.
- A silhouette alignment bug was exposed by an independent analytic footprint check, not by comparing a renderer to itself. The metric camera uses azimuth 90° with explicit pixel-origin alignment; final footprint IoUs are 0.901–0.991.
- Verified the upstream SAM 3D inference pipeline actually returns `output["glb"]`. Its README's Gaussian PLY example is not the only output format. The optional adapter exports the GLB directly; weights, GPU runtime and deployment remain unverified.
- Remote reconstruction caches include model/backend and stub-acceptance policy. A fake CPU-stub result cannot silently become an accepted production mesh. The primitive path remains the active simulation path.
- Budget caps and the USD 70 reserve cannot be relaxed by config. All production Meta clients share one global ledger; failed or unmetered attempts retain conservative charges. Cleanup failures retain active resource IDs and log only exception classes, not provider exception strings that could contain credentials.
- Cached synthetic artifacts bind immutable result/seal hashes. Core-source changes use a fresh synthetic revision instead of reopening an old one-shot seal. Real evaluation never automatically changes namespace or retries.
- The dashboard is a read-only native Streamlit viewer with local periodic refresh. Browser verification covered desktop/mobile layout, both actual videos, empty-result state and file-change refresh; no inference is triggered by viewing it.
- Exact installed dependencies are locked in `uv.lock`. Ruff formatting and focused annotations preserve deliberate stage-wide fallback and persisted-artifact validation boundaries instead of changing their public error contracts.
- After the user supplied the key, Wonder structured JSON, Spark image, Files upload and streamed SAM all passed on synthetic media; total Meta spend USD 0.028378, zero pending reservations. Official parser replay found two boxes/two masks. Raw credentials were not logged or published.
- Runpod access recheck: the live `xd://` registry exposes 11 devices, including AWS but no Runpod tools. This does not contradict the user's separate working connection. Native `/mcp reload`, `/mcp list` and `/mcp test <name>` can refresh/test this session. Official sources now recommend hosted `https://mcp.runpod.io/` or `npx @runpod/mcp-server@latest add`: https://docs.runpod.io/get-started/mcp-servers and https://github.com/runpod/runpod-mcp . No config was overwritten, provider substituted, or resource created.
- User-authorized setup found the working Runpod entry isolated in the research profile. Reused that entry in the default user MCP config, retained AWS unchanged, preserved the `!command` secure credential lookup, and restricted config permissions to 0600. An official MCP Python SDK stdio smoke exposed 68 tools and successfully called only `list-pods` (zero pods). The initial direct probe omitted the client's credential-command expansion and received 401; matching the documented resolver fixed the probe without changing the stored credential. No server reinstall, credential publication, resource mutation or GPU spend occurred. Current conversation tools require native `/mcp reload`.
