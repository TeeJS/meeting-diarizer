FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1 before other deps
RUN pip3 install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 10301

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10301", "--log-level", "info"]
