# vibe-tts Docker image with GPU support
# Build: docker build -t vibe-tts .
# Run (CPU): docker run -p 8000:8000 vibe-tts
# Run (GPU): docker run --gpus all -p 8000:8000 vibe-tts

FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install vibevoice from GitHub
RUN pip install --no-cache-dir git+https://github.com/microsoft/VibeVoice.git

# Copy project files
COPY pyproject.toml .
COPY vibe_tts/ vibe_tts/

# Install the package
RUN pip install --no-cache-dir .

# Pre-download the model (optional, makes container larger but faster startup)
# Uncomment to bake model into image:
# RUN python -c "from vibe_tts.model import load_model; load_model('vibe-1.5b', 'cpu', verbose=True)"

EXPOSE 8000

# Default command: start server on all interfaces
CMD ["vibe-tts", "serve", "--host", "0.0.0.0", "--port", "8000"]
