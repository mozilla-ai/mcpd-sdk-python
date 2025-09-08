.PHONY: ensure-scripts-exec setup test lint

ensure-scripts-exec:
	@chmod +x scripts/* || true

setup: ensure-scripts-exec
	@scripts/setup_uv.sh

test:
	@uv run $(DEBUGPY_ARGS) -m pytest tests

lint:
	@uv run pre-commit run --all-files
