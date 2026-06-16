# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    git supervisor curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Force uv to project virtual env mode inside the container
ENV UV_PROJECT_ENVIRONMENT=/venv \
    PATH="/venv/bin:$PATH" \
    AWS_ACCESS_KEY_ID=mock_key \
    AWS_SECRET_ACCESS_KEY=mock_secret \
    AWS_DEFAULT_REGION=us-east-1 \
    MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:5000 \
    DVC_S3_ENDPOINT_URL=http://127.0.0.1:5000

# Copy manifests and lockfile first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (cache enabled)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

COPY . /workspace/

# Install the project itself using uv sync (cache enabled)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Setup Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 5000 8000
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
