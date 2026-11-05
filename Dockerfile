# LLM-Firewall container image.
#
#   docker build -t llm-firewall .                                   # Layers A + B (~200 MB)
#   docker build -t llm-firewall:full --build-arg WITH_TRANSFORMER=true .   # + torch CPU for Layer C
#
# Layer C also needs a trained model in models/transformer/ at build time
# (or mounted at runtime: -v ./models/transformer:/app/models/transformer:ro).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG WITH_TRANSFORMER=false
COPY requirements.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_TRANSFORMER" = "true" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch \
      && pip install "transformers>=4.40" "tokenizers>=0.15"; \
    fi

# Application code + small artifacts only (see .dockerignore).
COPY llm_firewall ./llm_firewall
COPY config ./config
COPY models ./models

# Never run as root inside the container.
RUN useradd --create-home --uid 10001 firewall
USER firewall

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["uvicorn", "llm_firewall.proxy.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
