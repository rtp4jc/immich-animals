# Build from the repo root: docker build -f sidecar/Dockerfile -t animal-ml .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY sidecar/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sidecar/serve.py .
COPY models/onnx/detector.onnx models/onnx/embedding.onnx models/onnx/embedding.json models/onnx/

ENV MODEL_DIR=/app/models/onnx
EXPOSE 3003
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "3003"]
