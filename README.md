# Splatsam

Video-to-MuJoCo workspace reconstruction and honest grasp-planner evaluation.

Simulation only: a floating two-finger hand (no arm or inverse kinematics), and a rule-based planner with tunable parameters—not a trained neural policy. No real-robot transfer claims.

## Run offline

Install [uv](https://docs.astral.sh/uv/), GNU make and Mesa software rendering (Linux: libOSMesa). Then:

```sh
make setup && make demo
```

`make synth-demo` generates synthetic video, ground-truth mock segmentation, reconstruction, curriculum, validation-gated edits and a sealed synthetic test report. `make test` runs the behavioral suite. No model key/GPU is needed. Generated data lives in `data_synth/`; real video/private measurements in ignored `data/`.

Phase A stops at readiness. Real-video calls, the real seal and real final test require the user's separate `go` and action confirmations. GPU reconstruction is optional and additionally spend-gated. Never commit raw video or credentials.

API contracts and verification status: [docs/api_notes.md](docs/api_notes.md). Library choices: [docs/decisions.md](docs/decisions.md).
