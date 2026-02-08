#!/bin/bash
# ============================================================================
# DeepAgents - Podman Start Script
# Builds and runs the DeepAgents container using Podman
# ============================================================================

set -e

# Configuration
IMAGE_NAME="deepagents"
CONTAINER_NAME="deepagents-app"
HOST_PORT="${HOST_PORT:-8501}"
CONTAINER_PORT="8501"
DATA_DIR="${DATA_DIR:-./data}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "DeepAgents Podman Manager"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build       Build the container image"
    echo "  run         Run the container (builds if needed)"
    echo "  start       Start an existing stopped container"
    echo "  stop        Stop the running container"
    echo "  restart     Restart the container"
    echo "  logs        Show container logs"
    echo "  shell       Open a shell in the running container"
    echo "  clean       Remove container and image"
    echo "  status      Show container status"
    echo "  help        Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HOST_PORT       Host port to bind (default: 8501)"
    echo "  DATA_DIR        Directory for persistent data (default: ./data)"
    echo "  OLLAMA_BASE_URL Ollama server URL (default: http://host.containers.internal:11434)"
    echo "  OLLAMA_MODEL    Default Ollama model (default: llama3.2:latest)"
    echo ""
    echo "Examples:"
    echo "  $0 run                           # Build and run with defaults"
    echo "  HOST_PORT=9000 $0 run            # Run on port 9000"
    echo "  OLLAMA_BASE_URL=http://my-ollama:11434 $0 run"
    echo ""
}

check_podman() {
    if ! command -v podman &> /dev/null; then
        log_error "Podman is not installed. Please install Podman first."
        exit 1
    fi
}

build_image() {
    log_info "Building DeepAgents image..."
    podman build -t "${IMAGE_NAME}:latest" .
    log_success "Image built successfully: ${IMAGE_NAME}:latest"
}

stop_container() {
    if podman ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_info "Stopping container ${CONTAINER_NAME}..."
        podman stop "${CONTAINER_NAME}"
        log_success "Container stopped"
    fi
}

remove_container() {
    if podman ps -aq --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_info "Removing container ${CONTAINER_NAME}..."
        podman rm -f "${CONTAINER_NAME}"
        log_success "Container removed"
    fi
}

run_container() {
    # Check if image exists, build if not
    if ! podman images -q "${IMAGE_NAME}:latest" | grep -q .; then
        log_warning "Image not found. Building..."
        build_image
    fi

    # Stop and remove existing container
    remove_container

    # Create data directory if it doesn't exist
    mkdir -p "${DATA_DIR}"

    log_info "Starting DeepAgents container..."
    log_info "  Port: ${HOST_PORT}:${CONTAINER_PORT}"
    log_info "  Data: ${DATA_DIR}:/app/data"

    # Run the container
    podman run -d \
        --name "${CONTAINER_NAME}" \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        -v "$(realpath ${DATA_DIR}):/app/data:Z" \
        -e "OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.containers.internal:11434}" \
        -e "OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:latest}" \
        -e "OLLAMA_TIMEOUT=${OLLAMA_TIMEOUT:-120}" \
        -e "LOG_LEVEL=${LOG_LEVEL:-INFO}" \
        --restart unless-stopped \
        "${IMAGE_NAME}:latest"

    log_success "Container started successfully!"
    echo ""
    log_info "Access the application at: http://localhost:${HOST_PORT}"
    log_info "View logs with: $0 logs"
}

start_container() {
    if ! podman ps -aq --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_error "Container ${CONTAINER_NAME} does not exist. Use 'run' instead."
        exit 1
    fi

    log_info "Starting container ${CONTAINER_NAME}..."
    podman start "${CONTAINER_NAME}"
    log_success "Container started"
}

show_logs() {
    if ! podman ps -aq --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_error "Container ${CONTAINER_NAME} does not exist."
        exit 1
    fi
    podman logs -f "${CONTAINER_NAME}"
}

open_shell() {
    if ! podman ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_error "Container ${CONTAINER_NAME} is not running."
        exit 1
    fi
    log_info "Opening shell in ${CONTAINER_NAME}..."
    podman exec -it "${CONTAINER_NAME}" /bin/bash
}

clean_all() {
    log_warning "This will remove the container and image."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        remove_container
        if podman images -q "${IMAGE_NAME}:latest" | grep -q .; then
            log_info "Removing image ${IMAGE_NAME}:latest..."
            podman rmi "${IMAGE_NAME}:latest"
            log_success "Image removed"
        fi
        log_success "Cleanup complete"
    else
        log_info "Cleanup cancelled"
    fi
}

show_status() {
    echo ""
    echo "=== DeepAgents Container Status ==="
    echo ""

    if podman ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_success "Container is RUNNING"
        echo ""
        podman ps --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Status}}\t{{.Ports}}"
    elif podman ps -aq --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log_warning "Container exists but is STOPPED"
    else
        log_info "Container does not exist"
    fi

    echo ""
    echo "=== Image Status ==="
    if podman images -q "${IMAGE_NAME}:latest" | grep -q .; then
        podman images "${IMAGE_NAME}:latest" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.Created}}"
    else
        log_info "Image not built"
    fi
    echo ""
}

# Main script
check_podman

case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        start_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    clean)
        clean_all
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
