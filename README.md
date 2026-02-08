# DeepAgents Streamlit Explorer

A comprehensive Streamlit chat application for exploring the [LangChain DeepAgents](https://github.com/langchain-ai/deepagents) framework with local Ollama integration.

## Features

### Agent Types
Explore and compare different agent configurations:

| Agent Type | Description | Middleware |
|------------|-------------|------------|
| **Basic** | Minimal agent with core conversational capabilities | None |
| **Planning** | Task management with TodoListMiddleware | TodoList |
| **Filesystem** | Full file operation capabilities | TodoList, Filesystem |
| **SubAgent** | Multi-agent task delegation | TodoList, Filesystem, SubAgent |
| **Memory** | Persistent memory across sessions | TodoList, Memory, Filesystem |
| **Skills** | Progressive skill disclosure | TodoList, Skills, Filesystem |
| **Full** | All middleware enabled | All middleware |
| **Custom** | User-selected middleware stack | Configurable |

### Backend Options

- **StateBackend**: Ephemeral in-memory storage (per-thread)
- **StoreBackend**: Persistent cross-thread storage via LangGraph Store
- **FilesystemBackend**: Local disk storage with optional sandboxing
- **CompositeBackend**: Route paths to different backends
- **BaseSandbox**: Sandbox execution for shell commands

### Middleware Components

- **TodoListMiddleware**: Task planning and progress tracking
- **MemoryMiddleware**: Load AGENTS.md files into prompts
- **SkillsMiddleware**: Progressive skill disclosure via SKILL.md
- **FilesystemMiddleware**: File operations (ls, read, write, edit, glob, grep)
- **SubAgentMiddleware**: Task delegation to specialized subagents
- **SummarizationMiddleware**: Context window management
- **PatchToolCallsMiddleware**: Fix malformed tool calls
- **HumanInTheLoopMiddleware**: Approval gates for sensitive operations

### Dashboard Features

- Real-time interaction tracking
- Tool call monitoring with success rates
- Todo item tracking across agents
- Token usage analytics
- Duration metrics per agent type
- Historical interaction browser

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- Model: `qwen2.5:7b-instruct-q6_K` (or any Ollama model)

## Quick Start

### Windows (PowerShell)

```powershell
# Clone and navigate to the project
cd deepagents

# Start the application (auto-setup)
.\startup.ps1

# Or with options
.\startup.ps1 -InstallDeps -CheckOllama -PullModel
```

### Manual Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama serve

# Pull the model
ollama pull qwen2.5:7b-instruct-q6_K

# Run the application
streamlit run app.py
```

## Project Structure

```
deepagents/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── startup.ps1            # PowerShell startup script
├── .env.example           # Example environment variables
├── src/
│   ├── __init__.py
│   ├── agents.py          # Agent factory and configurations
│   ├── backends.py        # Backend implementations
│   ├── middleware.py      # Middleware configurations
│   ├── ollama_integration.py  # Ollama client and integration
│   └── history.py         # Interaction history tracking
└── deepagents_history.db  # SQLite database (created on first run)
```

## Configuration

Copy `.env.example` to `.env` and customize:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct-q6_K
STREAMLIT_PORT=8501
```

## Startup Script Options

```powershell
.\startup.ps1 [options]

Options:
    -InstallDeps    Install Python dependencies
    -CheckOllama    Verify Ollama connection
    -PullModel      Download the specified model
    -Model          Specify model (default: qwen2.5:7b-instruct-q6_K)
    -OllamaUrl      Ollama API URL (default: http://localhost:11434)
    -Port           Streamlit port (default: 8501)
    -Help           Show help message
```

## Usage

1. **Select Agent Type**: Choose from the sidebar dropdown
2. **Configure Backend**: Select storage backend for file operations
3. **Customize Middleware**: For Custom agents, select specific middleware
4. **Chat**: Interact with the agent in the chat interface
5. **Monitor**: View tool calls, todos, and metrics in real-time
6. **Analyze**: Use the History dashboard for analytics

## API Reference

### Creating Agents

```python
from src.agents import AgentFactory, AgentType
from src.ollama_integration import OllamaConfig

# Create agent configuration
config = AgentFactory.create_agent_config(
    agent_type=AgentType.FULL,
    ollama_config=OllamaConfig(model="qwen2.5:7b-instruct-q6_K"),
)

# Create the agent
agent = AgentFactory.create_agent(config)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

### Custom Backend

```python
from src.backends import BackendFactory, BackendType, BackendConfig

# Get all backend configs
backends = BackendFactory.get_all_backend_configs()

# Create specific backend
config = BackendConfig(
    backend_type=BackendType.COMPOSITE,
    name="MyBackend",
    description="Custom composite backend",
    options={
        "routes": {
            "/memories/": "store",
            "/workspace/": "filesystem"
        }
    }
)
backend = BackendFactory.create_backend(config)
```

### Custom Middleware

```python
from src.middleware import MiddlewareFactory, MiddlewareType

# Get all middleware configs
middleware = MiddlewareFactory.get_all_middleware_configs()

# Create custom stack
stack = MiddlewareFactory.create_middleware_stack([
    m for m in middleware
    if m.middleware_type in [
        MiddlewareType.TODOLIST,
        MiddlewareType.FILESYSTEM
    ]
])
```

## Sources

This application is built following the official LangChain DeepAgents documentation:

- [DeepAgents GitHub](https://github.com/langchain-ai/deepagents)
- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/)
- [DeepAgents API Reference](https://reference.langchain.com/python/deepagents/)
- [DeepAgents PyPI](https://pypi.org/project/deepagents/)
- [LangChain Middleware Architecture](https://www.blog.langchain.com/agent-middleware/)
- [DeepAgents Backends](https://docs.langchain.com/oss/python/deepagents/backends)

## License

MIT License - See LICENSE file for details.
