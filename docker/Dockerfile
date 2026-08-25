# Reproducible experiment environment (CUDA for Isaac Gym / torch)
FROM nvcr.io/nvidia/pytorch:24.04-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --extra sim

# Isaac Gym: mount or COPY the NVIDIA-distributed package at build/run time,
# then: uv pip install -e /opt/isaacgym/python
COPY configs/ configs/
COPY scripts/ scripts/
ENV PYTHONUNBUFFERED=1 PATH="/workspace/.venv/bin:$PATH"
CMD ["bash"]
