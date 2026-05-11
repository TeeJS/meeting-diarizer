FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with the matching CUDA wheel before other deps.
# The torch + nvidia-* wheels are 2-3 GB total and the nvidia mirror can
# be slow, so bump pip's default 60s timeout and add retries to keep the
# build from failing on transient read stalls.
RUN pip3 install --no-cache-dir --default-timeout=300 --retries=5 \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip3 install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt && \
    pip3 uninstall -y torchcodec || true

COPY app/ ./app/

EXPOSE 10301

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10301", "--log-level", "info"]
