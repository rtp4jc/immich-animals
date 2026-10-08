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
# Add --build-arg VARIANT=openvino (amd64 only) to run on an Intel GPU; the
# container then needs --device /dev/dri.
#
# Only the selected stages are built, so the other ones' inputs need not exist.
ARG MODEL_SOURCE=remote
ARG VARIANT=cpu
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

FROM python:3.12-slim AS runtime-cpu

FROM python:3.12-slim AS runtime-openvino
# Intel's OpenCL driver, which the OpenVINO GPU plugin talks to. Debian does not
# package it, so these are the debs Immich's own openvino image pins
# (machine-learning/scripts/install-intel-runtime.sh); the legacy pair covers
# GPUs older than Tiger Lake.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ocl-icd-libopencl1 curl \
    && mkdir /tmp/intel && cd /tmp/intel \
    && for url in \
        https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-core-2_2.41.5+22716_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/intel-opencl-icd_26.35.39758.10-0_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libigdgmm12_22.10.0_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb; \
       do curl -fsSLO "$url" || exit 1; done \
    && dpkg -i *.deb \
    && cd / && rm -rf /tmp/intel \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-${VARIANT}
ARG VARIANT

COPY --from=ghcr.io/astral-sh/uv:0.9.29 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY sidecar/pyproject.toml sidecar/uv.lock ./
RUN uv sync --frozen --no-default-groups --group ${VARIANT}

COPY sidecar/serve.py .
COPY --from=models /models/detector.onnx /models/embedding.onnx /models/embedding.json \
     /app/models/onnx/

ENV MODEL_DIR=/app/models/onnx \
    PATH=/app/.venv/bin:$PATH
EXPOSE 3003
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "3003"]
