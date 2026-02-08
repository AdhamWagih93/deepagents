# DeepAgents Streamlit Application Startup Script
# This script sets up and launches the DeepAgents Explorer application

param(
    [switch]$InstallDeps,
    [switch]$CheckOllama,
    [switch]$PullModel,
    [string]$Model = "qwen2.5:7b-instruct-q6_K",
    [string]$OllamaUrl = "http://localhost:11434",
    [int]$Port = 8501,
    [switch]$Help
)

# Colors for output
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    Write-ColorOutput "`n========================================" "Cyan"
    Write-ColorOutput "  DeepAgents Streamlit Explorer" "Cyan"
    Write-ColorOutput "========================================`n" "Cyan"
}

function Write-Step {
    param([string]$Step, [string]$Message)
    Write-ColorOutput "[$Step] $Message" "Yellow"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "[OK] $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "[ERROR] $Message" "Red"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "[INFO] $Message" "Gray"
}

# Show help
if ($Help) {
    Write-Header
    Write-Host @"
Usage: .\startup.ps1 [options]

Options:
    -InstallDeps    Install Python dependencies from requirements.txt
    -CheckOllama    Check if Ollama is running and accessible
    -PullModel      Pull the specified model from Ollama
    -Model          Specify the Ollama model (default: qwen2.5:7b-instruct-q6_K)
    -OllamaUrl      Ollama API URL (default: http://localhost:11434)
    -Port           Streamlit port (default: 8501)
    -Help           Show this help message

Examples:
    .\startup.ps1                          # Start the application
    .\startup.ps1 -InstallDeps             # Install dependencies and start
    .\startup.ps1 -CheckOllama -PullModel  # Check Ollama and pull model
    .\startup.ps1 -Port 8080               # Start on custom port

"@
    exit 0
}

Write-Header

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Info "Working directory: $ScriptDir"

# Step 1: Check Python installation
Write-Step "1/6" "Checking Python installation..."

$PythonCmd = $null
$PythonCommands = @("python", "python3", "py")

foreach ($cmd in $PythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($version -match "Python 3\.") {
            $PythonCmd = $cmd
            Write-Success "Found Python: $version"
            break
        }
    } catch {
        continue
    }
}

if (-not $PythonCmd) {
    Write-Error "Python 3 is not installed or not in PATH"
    Write-Info "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
}

# Step 2: Check/Create virtual environment
Write-Step "2/6" "Setting up virtual environment..."

$VenvPath = Join-Path $ScriptDir ".venv"
$VenvActivate = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $VenvPath)) {
    Write-Info "Creating virtual environment..."
    & $PythonCmd -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
    Write-Success "Virtual environment created at: $VenvPath"
    $InstallDeps = $true  # Force install deps for new venv
} else {
    Write-Success "Virtual environment exists: $VenvPath"
}

# Activate virtual environment
Write-Info "Activating virtual environment..."
. $VenvActivate

# Step 3: Install dependencies
if ($InstallDeps -or -not (Test-Path (Join-Path $VenvPath "Lib\site-packages\streamlit"))) {
    Write-Step "3/6" "Installing dependencies..."

    $RequirementsPath = Join-Path $ScriptDir "requirements.txt"

    if (Test-Path $RequirementsPath) {
        & pip install --upgrade pip
        & pip install -r $RequirementsPath

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install dependencies"
            exit 1
        }
        Write-Success "Dependencies installed successfully"
    } else {
        Write-Error "requirements.txt not found at: $RequirementsPath"
        exit 1
    }
} else {
    Write-Step "3/6" "Dependencies already installed (use -InstallDeps to reinstall)"
    Write-Success "Skipping dependency installation"
}

# Step 4: Check Ollama
Write-Step "4/6" "Checking Ollama connection..."

function Test-OllamaConnection {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri "$Url/api/tags" -Method Get -TimeoutSec 5 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

$OllamaRunning = Test-OllamaConnection -Url $OllamaUrl

if ($OllamaRunning) {
    Write-Success "Ollama is running at $OllamaUrl"

    # List available models
    try {
        $modelsResponse = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -Method Get
        $models = $modelsResponse.models | ForEach-Object { $_.name }
        Write-Info "Available models: $($models -join ', ')"
    } catch {
        Write-Info "Could not list models"
    }
} else {
    Write-Error "Ollama is not running at $OllamaUrl"
    Write-Info ""
    Write-Info "To start Ollama:"
    Write-Info "  1. Install Ollama from https://ollama.ai"
    Write-Info "  2. Run 'ollama serve' in a separate terminal"
    Write-Info "  3. Or start the Ollama application"
    Write-Info ""

    if (-not $CheckOllama) {
        Write-Info "Continuing anyway... (app will show disconnected status)"
    }
}

# Step 5: Pull model if requested
if ($PullModel -and $OllamaRunning) {
    Write-Step "5/6" "Pulling model: $Model..."

    try {
        # Check if model exists
        $modelsResponse = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -Method Get
        $modelExists = $modelsResponse.models | Where-Object { $_.name -like "$Model*" }

        if ($modelExists) {
            Write-Success "Model '$Model' is already available"
        } else {
            Write-Info "Downloading model '$Model'... (this may take a while)"

            # Start the pull process
            $pullBody = @{ name = $Model } | ConvertTo-Json
            $pullProcess = Start-Process -FilePath "ollama" -ArgumentList "pull", $Model -NoNewWindow -Wait -PassThru

            if ($pullProcess.ExitCode -eq 0) {
                Write-Success "Model '$Model' pulled successfully"
            } else {
                Write-Error "Failed to pull model"
            }
        }
    } catch {
        Write-Error "Failed to pull model: $_"
    }
} else {
    Write-Step "5/6" "Skipping model pull (use -PullModel to download)"
}

# Step 6: Start Streamlit application
Write-Step "6/6" "Starting Streamlit application..."

$AppPath = Join-Path $ScriptDir "app.py"

if (-not (Test-Path $AppPath)) {
    Write-Error "app.py not found at: $AppPath"
    exit 1
}

Write-ColorOutput "`n" "White"
Write-ColorOutput "========================================" "Green"
Write-ColorOutput "  Application Starting!" "Green"
Write-ColorOutput "========================================" "Green"
Write-ColorOutput "`n" "White"
Write-Info "URL: http://localhost:$Port"
Write-Info "Model: $Model"
Write-Info "Ollama: $OllamaUrl"
Write-ColorOutput "`n" "White"
Write-Info "Press Ctrl+C to stop the application"
Write-ColorOutput "`n" "White"

# Set environment variables
$env:OLLAMA_BASE_URL = $OllamaUrl
$env:OLLAMA_MODEL = $Model

# Start Streamlit
& streamlit run $AppPath --server.port $Port --server.headless true --browser.gatherUsageStats false

# Cleanup on exit
Write-Info "Application stopped."
