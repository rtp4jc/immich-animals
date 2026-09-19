# Build from the repo root: docker build -f sidecar/Dockerfile -t animal-ml .
# The context is the repo root only so the models can be copied in; nothing
# from the training project is installed.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.29 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY sidecar/pyproject.toml sidecar/uv.lock ./
RUN uv sync --frozen --no-dev

COPY sidecar/serve.py .
COPY models/onnx/detector.onnx models/onnx/embedding.onnx models/onnx/embedding.json models/onnx/

ENV MODEL_DIR=/app/models/onnx \
    PATH=/app/.venv/bin:$PATH
EXPOSE 3003
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "3003"]
