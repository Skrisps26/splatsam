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

Python 3.12 selected for scientific-wheel compatibility. Installed exact versions are recorded in uv.lock. Defaults use Standard Meta models rather than Contributor variants. No MODEL_API_KEY was present at project creation; the live spike remains explicitly blocked until supplied through the environment. Public GitHub requested; raw video/private data and secrets remain excluded. Scenario physics multipliers are hidden from planner's nominal mass/friction estimates.
