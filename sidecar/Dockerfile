# Build from the repo root. Models are not in git, so by default they are
# fetched from the release named by MODEL_TAG and checked against SHA256SUMS:
#
#   docker build -f sidecar/Dockerfile -t animal-ml .
#
# With the models on disk under models/onnx/<VERSION>/release/ (maintainers),
# skip the download:
#
#   docker build -f sidecar/Dockerfile --build-arg MODEL_SOURCE=local -t animal-ml .
#
# Only the selected stage is built, so the other one's inputs need not exist.
ARG MODEL_SOURCE=remote
ARG VERSION=0.3.0

FROM alpine:3.21 AS models-remote
ARG VERSION
ARG MODEL_TAG=sidecar-v${VERSION}
ARG MODEL_URL=https://github.com/rtp4jc/immich-animals/releases/download
WORKDIR /models
COPY sidecar/SHA256SUMS .
RUN for f in detector.onnx embedding.onnx embedding.json; do \
        wget -q "${MODEL_URL}/${MODEL_TAG}/$f" || exit 1; \
    done && sha256sum -c SHA256SUMS

FROM alpine:3.21 AS models-local
ARG VERSION
WORKDIR /models
COPY sidecar/SHA256SUMS .
COPY models/onnx/${VERSION}/release/detector.onnx models/onnx/${VERSION}/release/embedding.onnx models/onnx/${VERSION}/release/embedding.json ./
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

COPY sidecar/serve.py sidecar/preprocess.py ./
COPY --from=models /models/detector.onnx /models/embedding.onnx /models/embedding.json \
     /app/models/onnx/

ENV MODEL_DIR=/app/models/onnx \
    PATH=/app/.venv/bin:$PATH
RUN useradd --system --no-create-home --uid 10001 animal
USER animal
EXPOSE 3003
# Liveness on /, not /ping: /ping also reports upstream, and a restart does not
# bring a stopped upstream back.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3003/', timeout=4)"]
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "3003"]
