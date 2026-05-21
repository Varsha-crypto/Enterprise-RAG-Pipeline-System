FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=0
# Store HuggingFace models in a Docker volume so they persist across restarts
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA 12.1 support first (index takes priority over PyPI)
# Falls back to CPU automatically at runtime if no GPU is present
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
# Install remaining deps; torch is already installed so pip skips it
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 12000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "12000"]
