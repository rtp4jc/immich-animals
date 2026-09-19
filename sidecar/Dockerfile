# Build from the repo root. Models are not in git, so by default they are
# fetched from the release named by MODEL_TAG and checked against SHA256SUMS:
#
#   docker build -f sidecar/Dockerfile -t animal-ml .
#
# With the models on disk (maintainers), skip the download:
#
#   docker build -f sidecar/Dockerfile --build-arg MODEL_SOURCE=local -t animal-ml .
#
# Only the selected stage is built, so the other one's inputs need not exist.
ARG MODEL_SOURCE=remote

FROM alpine:3.21 AS models-remote
ARG MODEL_TAG=sidecar-v0.1.0
ARG MODEL_URL=https://github.com/rtp4jc/immich-animals/releases/download
WORKDIR /models
COPY sidecar/SHA256SUMS .
RUN for f in detector.onnx embedding.onnx embedding.json \
             embedding_resnet50.onnx embedding_resnet50.json; do \
        wget -q "${MODEL_URL}/${MODEL_TAG}/$f" || exit 1; \
    done && sha256sum -c SHA256SUMS

FROM alpine:3.21 AS models-local
WORKDIR /models
COPY sidecar/SHA256SUMS .
COPY models/onnx/detector.onnx models/onnx/embedding.onnx models/onnx/embedding.json \
     models/onnx/embedding_resnet50.onnx models/onnx/embedding_resnet50.json ./
RUN sha256sum -c SHA256SUMS

FROM models-${MODEL_SOURCE} AS models

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.29 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY sidecar/pyproject.toml sidecar/uv.lock ./
RUN uv sync --frozen --no-dev

COPY sidecar/serve.py .
COPY --from=models /models/detector.onnx /models/embedding.onnx /models/embedding.json \
     /models/embedding_resnet50.onnx /models/embedding_resnet50.json /app/models/onnx/

ENV MODEL_DIR=/app/models/onnx \
    PATH=/app/.venv/bin:$PATH
EXPOSE 3003
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "3003"]
