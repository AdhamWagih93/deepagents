# ============================================================================
# DeepAgents - Multistage Dockerfile
# LangChain Agent Explorer with Ollama Integration
# Includes: git, kubectl, podman CLIs for agent tools
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Production image with CLI tools
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Labels for container metadata
LABEL maintainer="DeepAgents Team" \
      description="LangChain Agent Explorer with Ollama Integration" \
      version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application defaults
    STREAMLIT_PORT=8501 \
    STREAMLIT_THEME=light \
    OLLAMA_BASE_URL=http://host.containers.internal:11434 \
    OLLAMA_MODEL=llama3.2:latest \
    OLLAMA_TIMEOUT=120 \
    HISTORY_DB_PATH=/app/data/deepagents_history.db \
    LOG_LEVEL=INFO \
    # Streamlit configuration
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Kubernetes config
    KUBECONFIG=/app/.kube/config

# Install runtime dependencies and CLI tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    gnupg \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' > /etc/apt/sources.list.d/kubernetes.list && \
    apt-get update && apt-get install -y kubectl && \
    rm -rf /var/lib/apt/lists/*

# Install podman (podman-remote for connecting to external podman)
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/unstable/Debian_12/Release.key | gpg --dearmor -o /etc/apt/keyrings/libcontainers.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/libcontainers.gpg] https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/unstable/Debian_12/ /" > /etc/apt/sources.list.d/libcontainers.list && \
    apt-get update && apt-get install -y podman-remote && \
    ln -s /usr/bin/podman-remote /usr/bin/podman && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 deepagents && \
    useradd --uid 1000 --gid deepagents --shell /bin/bash --create-home deepagents

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=deepagents:deepagents . .

# Create directories for persistent storage and configs
RUN mkdir -p /app/data /app/.kube /app/.ssh && \
    chown -R deepagents:deepagents /app/data /app/.kube /app/.ssh

# Switch to non-root user
USER deepagents

# Configure git for the user
RUN git config --global init.defaultBranch main && \
    git config --global user.email "deepagents@local" && \
    git config --global user.name "DeepAgents"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: Run Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
