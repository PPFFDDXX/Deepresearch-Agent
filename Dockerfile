FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid agent --create-home agent \
    && mkdir -p /app/reports \
    && chown agent:agent /app/reports

# Explicit copies keep local credentials and generated reports out of the image.
# Include tests and fixtures so the same image supports offline verification.
COPY --chown=agent:agent app/ ./app/
COPY --chown=agent:agent examples/ ./examples/
COPY --chown=agent:agent evals/ ./evals/
COPY --chown=agent:agent tests/ ./tests/
COPY --chown=agent:agent main.py ./

USER agent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); assert json.load(response)['status'] == 'ok'"]

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
