"""
LangChain Agent Framework

A pure LangChain implementation for building agents with Ollama.
Uses built-in LangChain middleware from langchain.agents.middleware.

Based on working implementation from sample_agent.py.

Modules:
- agents: Agent factory using create_agent from langchain.agents
- middleware: 14 built-in middleware types from LangChain
- ollama_integration: Ollama LLM integration via ChatOllama
- backends: Checkpointing and state management
- history: Interaction tracking and storage
- debug: Logging and debugging utilities
"""

__version__ = "1.0.0"

from .agents import (
    AgentConfig,
    AgentFactory,
    LangChainAgent,
    AgentInteraction,
    AgentSession,
)

from .middleware import (
    MiddlewareType,
    MiddlewareConfig,
    MiddlewareFactory,
    MiddlewareStack,
    MIDDLEWARE_CONFIGS,
    MIDDLEWARE_PRESETS,
    # Built-in LangChain middleware (re-exported)
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    LLMToolEmulator,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ShellToolMiddleware,
    HostExecutionPolicy,
    DockerExecutionPolicy,
    FilesystemFileSearchMiddleware,
)

from .ollama_integration import (
    OllamaConfig,
    OllamaClient,
    OllamaMetrics,
    create_ollama_llm,
    OLLAMA_PRESETS,
    test_ollama_connection,
    test_model_availability,
    test_model_inference,
)

from .backends import (
    BackendType,
    BackendConfig,
    BackendFactory,
    BackendMetrics,
)

from .history import (
    HistoryStore,
    InteractionTracker,
    InteractionRecord,
    ToolCallRecord,
    TodoRecord,
)

from .debug import (
    DebugLevel,
    AgentPhase,
    DebugEvent,
    DebugLogger,
    get_debug_logger,
    set_debug_level,
)

__all__ = [
    # Agents
    "AgentConfig",
    "AgentFactory",
    "LangChainAgent",
    "AgentInteraction",
    "AgentSession",
    # Middleware
    "MiddlewareType",
    "MiddlewareConfig",
    "MiddlewareFactory",
    "MiddlewareStack",
    "MIDDLEWARE_CONFIGS",
    "MIDDLEWARE_PRESETS",
    # Built-in LangChain middleware
    "SummarizationMiddleware",
    "HumanInTheLoopMiddleware",
    "ModelCallLimitMiddleware",
    "ToolCallLimitMiddleware",
    "ModelFallbackMiddleware",
    "PIIMiddleware",
    "TodoListMiddleware",
    "LLMToolSelectorMiddleware",
    "ToolRetryMiddleware",
    "ModelRetryMiddleware",
    "LLMToolEmulator",
    "ContextEditingMiddleware",
    "ClearToolUsesEdit",
    "ShellToolMiddleware",
    "HostExecutionPolicy",
    "DockerExecutionPolicy",
    "FilesystemFileSearchMiddleware",
    # Ollama
    "OllamaConfig",
    "OllamaClient",
    "OllamaMetrics",
    "create_ollama_llm",
    "OLLAMA_PRESETS",
    "test_ollama_connection",
    "test_model_availability",
    "test_model_inference",
    # Backends
    "BackendType",
    "BackendConfig",
    "BackendFactory",
    "BackendMetrics",
    # History
    "HistoryStore",
    "InteractionTracker",
    "InteractionRecord",
    "ToolCallRecord",
    "TodoRecord",
    # Debug
    "DebugLevel",
    "AgentPhase",
    "DebugEvent",
    "DebugLogger",
    "get_debug_logger",
    "set_debug_level",
]
