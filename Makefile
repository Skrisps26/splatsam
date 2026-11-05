SHELL := /bin/bash
.PHONY: setup test synth-demo demo export-ui final-test
setup:
	uv sync --python 3.12
	mkdir -p data/videos

test:
	uv run pytest -q

synth-demo:
	uv run python scripts/run_pipeline.py --mode synthetic --root data_synth --run-id synth-demo --seed 42
	$(MAKE) export-ui RUN_DIR=runs/synth-demo

demo:
	uv run python scripts/run_pipeline.py --mode cached --run-id demo
	uv run streamlit run dashboard/app.py --server.address 127.0.0.1 -- --run-dir runs/demo

export-ui:
	@test -n "$(RUN_DIR)" || (echo 'RUN_DIR required; exports existing artifacts only'; exit 2)
	uv run python scripts/export_ui_media.py --run-dir "$(RUN_DIR)" --bundle "$(RUN_DIR)"
	uv run python scripts/export_ui_events.py --run-dir "$(RUN_DIR)" --destination "$(RUN_DIR)"

final-test:
	@test -n "$(RUN_DIR)" -a -n "$(DATA_ROOT)" || (echo 'RUN_DIR and DATA_ROOT required'; exit 2)
	uv run python scripts/run_final_test.py --run-dir "$(RUN_DIR)" --root "$(DATA_ROOT)"
