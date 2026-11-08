# 1. Base image (pyproject requires >=3.12)
FROM python:3.12-slim

# 2. Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# 3. Working directory
WORKDIR /app

# 4. System dependencies (OpenCV runtime + curl for weights/healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 5. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 6. Copy dependency definitions + app source
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# 7. Install Python dependencies
RUN uv sync --frozen --no-dev

# 8. Download model weights from a GitHub Release.
#    Override on model releases: --build-arg WEIGHTS_URL=.../roof-vX.Y.Z/best.pt
ARG WEIGHTS_URL="https://github.com/rafidtya/sunsight-ml/releases/latest/download/best.pt"
RUN mkdir -p src/roof_inference/weights \
    && curl -fsSL "$WEIGHTS_URL" -o src/roof_inference/weights/best.pt \
    && test -s src/roof_inference/weights/best.pt

# 9. Configure Python path
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# 10. Document application port
EXPOSE 8000

# 11. Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

# 12. Start application
CMD ["uvicorn", "roof_inference.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]