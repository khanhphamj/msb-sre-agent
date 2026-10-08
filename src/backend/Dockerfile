# AgentBase Runtime: listen 0.0.0.0:8080, GET /health -> 200. Build for linux/amd64.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependency layer (cache-friendly): copy only the lockfile first
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 agent && chown -R agent /app
USER agent

EXPOSE 8080
CMD ["python", "main.py"]
