# ============================================================================
# DeepAgents - Docker Startup Script for Windows
# Builds and runs the DeepAgents container using Docker
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "run", "start", "stop", "restart", "logs", "shell", "clean", "status", "help")]
    [string]$Command = "help",

    [int]$HostPort = 8501,
    [string]$DataDir = ".\data",
    [string]$KubeConfig = "$env:USERPROFILE\.kube\config",
    [string]$OllamaUrl = "http://host.docker.internal:11434",
    [string]$OllamaModel = "llama3.2:latest"
)

# Configuration
$ImageName = "deepagents"
$ContainerName = "deepagents-app"
$ContainerPort = 8501

# Colors for output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Show-Help {
    Write-Host @"
DeepAgents Docker Manager for Windows

Usage: .\docker_startup.ps1 <command> [options]

Commands:
  build       Build the container image
  run         Run the container (builds if needed)
  start       Start an existing stopped container
  stop        Stop the running container
  restart     Restart the container
  logs        Show container logs (follow mode)
  shell       Open a shell in the running container
  clean       Remove container and image
  status      Show container status
  help        Show this help message

Options:
  -HostPort      Host port to bind (default: 8501)
  -DataDir       Directory for persistent data (default: .\data)
  -KubeConfig    Path to kubeconfig file (default: ~\.kube\config)
  -OllamaUrl     Ollama server URL (default: http://host.docker.internal:11434)
  -OllamaModel   Default Ollama model (default: llama3.2:latest)

Examples:
  .\docker_startup.ps1 run
  .\docker_startup.ps1 run -HostPort 9000
  .\docker_startup.ps1 run -OllamaUrl "http://my-server:11434"
  .\docker_startup.ps1 logs
  .\docker_startup.ps1 shell

"@
}

function Test-Docker {
    try {
        $null = docker version 2>&1
        return $true
    } catch {
        Write-Error "Docker is not installed or not running. Please install Docker Desktop."
        return $false
    }
}

function Build-Image {
    Write-Info "Building DeepAgents image..."
    docker build -t "${ImageName}:latest" .
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Image built successfully: ${ImageName}:latest"
    } else {
        Write-Error "Failed to build image"
        exit 1
    }
}

function Stop-Container {
    $running = docker ps -q --filter "name=$ContainerName"
    if ($running) {
        Write-Info "Stopping container $ContainerName..."
        docker stop $ContainerName
        Write-Success "Container stopped"
    }
}

function Remove-Container {
    $exists = docker ps -aq --filter "name=$ContainerName"
    if ($exists) {
        Write-Info "Removing container $ContainerName..."
        docker rm -f $ContainerName
        Write-Success "Container removed"
    }
}

function Start-NewContainer {
    # Check if image exists, build if not
    $imageExists = docker images -q "${ImageName}:latest"
    if (-not $imageExists) {
        Write-Warning "Image not found. Building..."
        Build-Image
    }

    # Stop and remove existing container
    Remove-Container

    # Create data directory if it doesn't exist
    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    }
    $DataDirFull = (Resolve-Path $DataDir).Path

    # Prepare volume mounts
    $volumes = @(
        "-v", "${DataDirFull}:/app/data:rw"
    )

    # Mount kubeconfig if it exists
    if (Test-Path $KubeConfig) {
        $KubeConfigFull = (Resolve-Path $KubeConfig).Path
        $volumes += "-v"
        $volumes += "${KubeConfigFull}:/app/.kube/config:ro"
        Write-Info "Mounting kubeconfig from: $KubeConfigFull"
    } else {
        Write-Warning "Kubeconfig not found at $KubeConfig - kubectl will not have access to cluster"
    }

    # Mount SSH directory if it exists (for git operations)
    $sshDir = "$env:USERPROFILE\.ssh"
    if (Test-Path $sshDir) {
        $volumes += "-v"
        $volumes += "${sshDir}:/app/.ssh:ro"
        Write-Info "Mounting SSH keys from: $sshDir"
    }

    Write-Info "Starting DeepAgents container..."
    Write-Info "  Port: ${HostPort}:${ContainerPort}"
    Write-Info "  Data: $DataDirFull"
    Write-Info "  Ollama: $OllamaUrl"

    # Run the container
    $dockerArgs = @(
        "run", "-d",
        "--name", $ContainerName,
        "-p", "${HostPort}:${ContainerPort}"
    ) + $volumes + @(
        "-e", "OLLAMA_BASE_URL=$OllamaUrl",
        "-e", "OLLAMA_MODEL=$OllamaModel",
        "-e", "OLLAMA_TIMEOUT=120",
        "-e", "LOG_LEVEL=INFO",
        "--restart", "unless-stopped",
        "${ImageName}:latest"
    )

    docker @dockerArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Container started successfully!"
        Write-Host ""
        Write-Info "Access the application at: http://localhost:${HostPort}"
        Write-Info "View logs with: .\docker_startup.ps1 logs"
    } else {
        Write-Error "Failed to start container"
        exit 1
    }
}

function Start-ExistingContainer {
    $exists = docker ps -aq --filter "name=$ContainerName"
    if (-not $exists) {
        Write-Error "Container $ContainerName does not exist. Use 'run' instead."
        exit 1
    }

    Write-Info "Starting container $ContainerName..."
    docker start $ContainerName
    Write-Success "Container started"
}

function Show-Logs {
    $exists = docker ps -aq --filter "name=$ContainerName"
    if (-not $exists) {
        Write-Error "Container $ContainerName does not exist."
        exit 1
    }
    docker logs -f $ContainerName
}

function Open-Shell {
    $running = docker ps -q --filter "name=$ContainerName"
    if (-not $running) {
        Write-Error "Container $ContainerName is not running."
        exit 1
    }
    Write-Info "Opening shell in $ContainerName..."
    docker exec -it $ContainerName /bin/bash
}

function Remove-All {
    Write-Warning "This will remove the container and image."
    $confirm = Read-Host "Are you sure? (y/N)"
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Remove-Container
        $imageExists = docker images -q "${ImageName}:latest"
        if ($imageExists) {
            Write-Info "Removing image ${ImageName}:latest..."
            docker rmi "${ImageName}:latest"
            Write-Success "Image removed"
        }
        Write-Success "Cleanup complete"
    } else {
        Write-Info "Cleanup cancelled"
    }
}

function Show-Status {
    Write-Host ""
    Write-Host "=== DeepAgents Container Status ===" -ForegroundColor Cyan
    Write-Host ""

    $running = docker ps -q --filter "name=$ContainerName"
    $exists = docker ps -aq --filter "name=$ContainerName"

    if ($running) {
        Write-Success "Container is RUNNING"
        Write-Host ""
        docker ps --filter "name=$ContainerName" --format "table {{.ID}}\t{{.Status}}\t{{.Ports}}"
    } elseif ($exists) {
        Write-Warning "Container exists but is STOPPED"
    } else {
        Write-Info "Container does not exist"
    }

    Write-Host ""
    Write-Host "=== Image Status ===" -ForegroundColor Cyan
    $imageExists = docker images -q "${ImageName}:latest"
    if ($imageExists) {
        docker images "${ImageName}:latest" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
    } else {
        Write-Info "Image not built"
    }

    # Show CLI versions in container if running
    if ($running) {
        Write-Host ""
        Write-Host "=== CLI Tools in Container ===" -ForegroundColor Cyan
        docker exec $ContainerName git --version 2>$null
        docker exec $ContainerName kubectl version --client --short 2>$null
        docker exec $ContainerName podman --version 2>$null
    }

    Write-Host ""
}

# Main execution
if (-not (Test-Docker)) {
    exit 1
}

switch ($Command) {
    "build"   { Build-Image }
    "run"     { Start-NewContainer }
    "start"   { Start-ExistingContainer }
    "stop"    { Stop-Container }
    "restart" { Stop-Container; Start-ExistingContainer }
    "logs"    { Show-Logs }
    "shell"   { Open-Shell }
    "clean"   { Remove-All }
    "status"  { Show-Status }
    "help"    { Show-Help }
    default   { Show-Help }
}
