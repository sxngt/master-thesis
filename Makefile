# All tooling runs through uv (https://docs.astral.sh/uv/).
.PHONY: install install-sim test lint format smoke lock clean

install:
	uv sync

install-sim:
	uv sync --extra sim
	@echo "NOTE: Isaac Gym must be installed manually from NVIDIA:"
	@echo "  uv pip install -e <isaacgym>/python   (see docs/setup.md)"

lock:
	uv lock

test:
	uv run pytest tests/

lint:
	uv run ruff check src/ scripts/ tests/
	uv run ruff format --check src/ scripts/ tests/

format:
	uv run ruff format src/ scripts/ tests/
	uv run ruff check --fix src/ scripts/ tests/

# 1-minute end-to-end sanity check (mock backend, no simulator required)
smoke:
	uv run python scripts/train.py --sim mock --algorithm ppo --robot a1 --terrain flat --seed 0 --smoke-test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info
