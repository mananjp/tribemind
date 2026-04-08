# ─────────────────────────────────────────────────────────────
# TribeMind — Single-container Docker build
# Runs: FastAPI backend + Streamlit frontend + Nginx reverse proxy
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
# Install CPU-only PyTorch first (smaller footprint)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install backend-specific deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    huggingface_hub \
    transformers \
    scipy \
    opencv-python-headless \
    python-dotenv

# ── Application code ─────────────────────────────────────────
COPY app.py .
COPY server.py .
COPY inference.py .
COPY brain_regions.py .
COPY .env.example .env
COPY .streamlit/ .streamlit/

# ── Nginx config ─────────────────────────────────────────────
RUN rm -f /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/tribemind.conf

# ── Supervisord config ───────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Healthcheck ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Expose the single nginx port
EXPOSE 80

# The TRIBE_BACKEND_URL inside the container points to itself
ENV TRIBE_BACKEND_URL=http://127.0.0.1:8000
ENV PYTHONUNBUFFERED=1

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
