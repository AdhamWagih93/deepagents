"""
LangChain Built-in Middleware

This module provides configuration and helpers for LangChain's built-in middleware components.
All middleware is imported directly from langchain.agents.middleware.

Based on working implementation from sample_agent.py.

Built-in middleware types (14 total):
1. SummarizationMiddleware - Auto-summarize conversation history
2. HumanInTheLoopMiddleware - Pause for human approval
3. ModelCallLimitMiddleware - Limit model API calls
4. ToolCallLimitMiddleware - Limit tool executions
5. ModelFallbackMiddleware - Fallback to alternative models
6. PIIMiddleware - Detect and handle PII
7. TodoListMiddleware - Task planning and tracking
8. LLMToolSelectorMiddleware - Pre-select relevant tools
9. ToolRetryMiddleware - Retry failed tool calls
10. ModelRetryMiddleware - Retry failed model calls
11. LLMToolEmulator - Emulate tools for testing
12. ContextEditingMiddleware - Manage context window
13. ShellToolMiddleware - Persistent shell session
14. FilesystemFileSearchMiddleware - Glob and Grep search
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import all built-in middleware from LangChain
from langchain.agents.middleware import (
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


class MiddlewareType(Enum):
    """Available middleware types."""
    SUMMARIZATION = "summarization"
    HUMAN_IN_LOOP = "human_in_loop"
    MODEL_CALL_LIMIT = "model_call_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    MODEL_FALLBACK = "model_fallback"
    PII_DETECTION = "pii_detection"
    TODOLIST = "todolist"
    LLM_TOOL_SELECTOR = "llm_tool_selector"
    TOOL_RETRY = "tool_retry"
    MODEL_RETRY = "model_retry"
    LLM_TOOL_EMULATOR = "llm_tool_emulator"
    CONTEXT_EDITING = "context_editing"
    SHELL_TOOL = "shell_tool"
    FILE_SEARCH = "file_search"
    SECRET_INJECTION = "secret_injection"


class SecretInjectionMiddleware:
    """
    Custom middleware for injecting secrets into tool execution.

    This middleware:
    1. Pre-execution: Loads required secrets into environment variables
    2. Post-execution: Cleans up secrets and masks them in output logs
    """

    def __init__(self, secrets_manager=None, secret_names: List[str] = None):
        """
        Initialize the SecretInjectionMiddleware.

        Args:
            secrets_manager: SecretsManager instance for secret access
            secret_names: List of secret names to inject
        """
        self.secrets_manager = secrets_manager
        self.secret_names = secret_names or []
        self._injected_secrets = {}
        self._original_env = {}

    def pre_execute(self):
        """Inject secrets before tool execution."""
        import os

        if not self.secrets_manager:
            return

        for name in self.secret_names:
            value = self.secrets_manager.get_secret(name)
            if value:
                # Store original value
                self._original_env[name] = os.environ.get(name)
                # Inject secret
                os.environ[name] = value
                self._injected_secrets[name] = value

    def post_execute(self, output: str = "") -> str:
        """Clean up secrets and mask in output."""
        import os

        # Restore original environment
        for name in self._injected_secrets:
            if self._original_env.get(name) is not None:
                os.environ[name] = self._original_env[name]
            else:
                os.environ.pop(name, None)

        # Mask secrets in output
        masked_output = output
        if self.secrets_manager:
            masked_output = self.secrets_manager.mask_in_logs(output)

        # Clear tracking
        self._injected_secrets = {}
        self._original_env = {}

        return masked_output

    def wrap_tool_call(self, tool_func, *args, **kwargs):
        """Wrap a tool call with secret injection."""
        self.pre_execute()
        try:
            result = tool_func(*args, **kwargs)
            if isinstance(result, str):
                result = self.post_execute(result)
            else:
                self.post_execute()
            return result
        except Exception as e:
            self.post_execute()
            raise


@dataclass
class MiddlewareConfig:
    """Configuration for a middleware instance."""
    middleware_type: MiddlewareType
    name: str
    description: str
    enabled: bool = True
    priority: int = 0
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.middleware_type.value,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "priority": self.priority,
            "options": self.options
        }


# Middleware configurations with default options based on sample_agent.py
MIDDLEWARE_CONFIGS: Dict[MiddlewareType, MiddlewareConfig] = {
    MiddlewareType.SUMMARIZATION: MiddlewareConfig(
        middleware_type=MiddlewareType.SUMMARIZATION,
        name="SummarizationMiddleware",
        description="Automatically summarize conversation history when approaching token limits.",
        priority=1,
        options={
            "model": None,  # Required: LLM for summarization
            "trigger": ("tokens", 20000),  # Trigger on token count
            "keep": ("messages", 20),  # Keep last 20 messages
        }
    ),
    MiddlewareType.HUMAN_IN_LOOP: MiddlewareConfig(
        middleware_type=MiddlewareType.HUMAN_IN_LOOP,
        name="HumanInTheLoopMiddleware",
        description="Pause agent execution for human approval, editing, or rejection of tool calls.",
        priority=2,
        options={
            "interrupt_on": {},  # Dict: tool_name -> {allowed_decisions: [...]} or True
        }
    ),
    MiddlewareType.MODEL_CALL_LIMIT: MiddlewareConfig(
        middleware_type=MiddlewareType.MODEL_CALL_LIMIT,
        name="ModelCallLimitMiddleware",
        description="Prevent excessive model API calls and control costs.",
        priority=3,
        options={
            "thread_limit": 10,
            "run_limit": 5,
            "exit_behavior": "end",  # 'end' or 'error'
        }
    ),
    MiddlewareType.TOOL_CALL_LIMIT: MiddlewareConfig(
        middleware_type=MiddlewareType.TOOL_CALL_LIMIT,
        name="ToolCallLimitMiddleware",
        description="Control agent execution by limiting the number of tool calls.",
        priority=4,
        options={
            "tool_name": None,  # Optional: limit specific tool
            "thread_limit": 20,
            "run_limit": 10,
        }
    ),
    MiddlewareType.MODEL_FALLBACK: MiddlewareConfig(
        middleware_type=MiddlewareType.MODEL_FALLBACK,
        name="ModelFallbackMiddleware",
        description="Automatically fallback to alternative models when primary fails.",
        priority=5,
        options={
            "fallback_models": [],  # List of ChatOllama/LLM instances
        }
    ),
    MiddlewareType.PII_DETECTION: MiddlewareConfig(
        middleware_type=MiddlewareType.PII_DETECTION,
        name="PIIMiddleware",
        description="Detect and handle Personally Identifiable Information (PII).",
        priority=6,
        options={
            "pii_type": "email",  # email, credit_card, ip, mac_address, url
            "strategy": "redact",  # block, redact, mask, hash
            "apply_to_input": True,
        }
    ),
    MiddlewareType.TODOLIST: MiddlewareConfig(
        middleware_type=MiddlewareType.TODOLIST,
        name="TodoListMiddleware",
        description="Equip agents with task planning and tracking capabilities. Adds write_todos tool.",
        priority=7,
        options={}
    ),
    MiddlewareType.LLM_TOOL_SELECTOR: MiddlewareConfig(
        middleware_type=MiddlewareType.LLM_TOOL_SELECTOR,
        name="LLMToolSelectorMiddleware",
        description="Use an LLM to select relevant tools before calling main model.",
        priority=8,
        options={
            "model": None,  # Required: LLM for selection
            "max_tools": 5,
            "always_include": [],  # Tool names to always include
        }
    ),
    MiddlewareType.TOOL_RETRY: MiddlewareConfig(
        middleware_type=MiddlewareType.TOOL_RETRY,
        name="ToolRetryMiddleware",
        description="Automatically retry failed tool calls with exponential backoff.",
        priority=9,
        options={
            "max_retries": 3,
            "backoff_factor": 2.0,
            "initial_delay": 1.0,
            "tools": None,  # None = all tools, or list of specific tools
        }
    ),
    MiddlewareType.MODEL_RETRY: MiddlewareConfig(
        middleware_type=MiddlewareType.MODEL_RETRY,
        name="ModelRetryMiddleware",
        description="Automatically retry failed model calls with exponential backoff.",
        priority=10,
        options={
            "max_retries": 3,
            "backoff_factor": 2.0,
            "initial_delay": 1.0,
        }
    ),
    MiddlewareType.LLM_TOOL_EMULATOR: MiddlewareConfig(
        middleware_type=MiddlewareType.LLM_TOOL_EMULATOR,
        name="LLMToolEmulator",
        description="Emulate tool execution using an LLM for testing purposes.",
        priority=11,
        options={
            "model": None,  # Required: LLM for emulation
            "tools": [],  # Tools to emulate
        }
    ),
    MiddlewareType.CONTEXT_EDITING: MiddlewareConfig(
        middleware_type=MiddlewareType.CONTEXT_EDITING,
        name="ContextEditingMiddleware",
        description="Manage conversation context by clearing older tool call outputs.",
        priority=12,
        options={
            "trigger": 10000,  # Token count trigger
            "keep": 5,  # Keep last N tool uses
            "exclude_tools": [],  # Tools to exclude from clearing
            "placeholder": "[cleared]",  # Replacement text
        }
    ),
    MiddlewareType.SHELL_TOOL: MiddlewareConfig(
        middleware_type=MiddlewareType.SHELL_TOOL,
        name="ShellToolMiddleware",
        description="Expose a persistent shell session to agents for command execution.",
        priority=13,
        options={
            "workspace_root": ".",
            "execution_policy": "host",  # host or docker
        }
    ),
    MiddlewareType.FILE_SEARCH: MiddlewareConfig(
        middleware_type=MiddlewareType.FILE_SEARCH,
        name="FilesystemFileSearchMiddleware",
        description="Provide Glob and Grep search tools over a filesystem.",
        priority=14,
        options={
            "root_path": ".",
            "use_ripgrep": True,
            "max_file_size_mb": 10,
        }
    ),
    MiddlewareType.SECRET_INJECTION: MiddlewareConfig(
        middleware_type=MiddlewareType.SECRET_INJECTION,
        name="SecretInjectionMiddleware",
        description="Inject secrets into environment for tool execution and mask in output.",
        priority=15,
        options={
            "secrets_manager": None,  # SecretsManager instance
            "secret_names": [],  # List of secret names to inject
        }
    ),
}


class MiddlewareFactory:
    """Factory for creating LangChain middleware instances."""

    @staticmethod
    def get_all_middleware_configs() -> List[MiddlewareConfig]:
        """Return all available middleware configurations."""
        return list(MIDDLEWARE_CONFIGS.values())

    @staticmethod
    def create_middleware(
        middleware_type: MiddlewareType,
        model: Any = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a LangChain middleware instance.

        Args:
            middleware_type: Type of middleware to create
            model: LLM model (required for some middleware)
            options: Configuration options (merged with defaults)

        Returns:
            LangChain middleware instance
        """
        config = MIDDLEWARE_CONFIGS.get(middleware_type)
        if not config:
            raise ValueError(f"Unknown middleware type: {middleware_type}")

        # Merge options with defaults
        merged_options = {**config.options, **(options or {})}

        if middleware_type == MiddlewareType.SUMMARIZATION:
            return SummarizationMiddleware(
                model=model or merged_options.get("model"),
                trigger=merged_options.get("trigger", ("tokens", 20000)),
                keep=merged_options.get("keep", ("messages", 20)),
            )

        elif middleware_type == MiddlewareType.HUMAN_IN_LOOP:
            return HumanInTheLoopMiddleware(
                interrupt_on=merged_options.get("interrupt_on", {}),
            )

        elif middleware_type == MiddlewareType.MODEL_CALL_LIMIT:
            return ModelCallLimitMiddleware(
                thread_limit=merged_options.get("thread_limit", 10),
                run_limit=merged_options.get("run_limit", 5),
                exit_behavior=merged_options.get("exit_behavior", "end"),
            )

        elif middleware_type == MiddlewareType.TOOL_CALL_LIMIT:
            kwargs = {
                "thread_limit": merged_options.get("thread_limit", 20),
                "run_limit": merged_options.get("run_limit", 10),
            }
            if merged_options.get("tool_name"):
                kwargs["tool_name"] = merged_options["tool_name"]
            return ToolCallLimitMiddleware(**kwargs)

        elif middleware_type == MiddlewareType.MODEL_FALLBACK:
            fallback_models = merged_options.get("fallback_models", [])
            return ModelFallbackMiddleware(*fallback_models)

        elif middleware_type == MiddlewareType.PII_DETECTION:
            return PIIMiddleware(
                merged_options.get("pii_type", "email"),
                strategy=merged_options.get("strategy", "redact"),
                apply_to_input=merged_options.get("apply_to_input", True),
            )

        elif middleware_type == MiddlewareType.TODOLIST:
            return TodoListMiddleware()

        elif middleware_type == MiddlewareType.LLM_TOOL_SELECTOR:
            return LLMToolSelectorMiddleware(
                model=model or merged_options.get("model"),
                max_tools=merged_options.get("max_tools", 5),
                always_include=merged_options.get("always_include", []),
            )

        elif middleware_type == MiddlewareType.TOOL_RETRY:
            kwargs = {
                "max_retries": merged_options.get("max_retries", 3),
                "backoff_factor": merged_options.get("backoff_factor", 2.0),
                "initial_delay": merged_options.get("initial_delay", 1.0),
            }
            if merged_options.get("tools"):
                kwargs["tools"] = merged_options["tools"]
            return ToolRetryMiddleware(**kwargs)

        elif middleware_type == MiddlewareType.MODEL_RETRY:
            return ModelRetryMiddleware(
                max_retries=merged_options.get("max_retries", 3),
                backoff_factor=merged_options.get("backoff_factor", 2.0),
                initial_delay=merged_options.get("initial_delay", 1.0),
            )

        elif middleware_type == MiddlewareType.LLM_TOOL_EMULATOR:
            return LLMToolEmulator(
                model=model or merged_options.get("model"),
                tools=merged_options.get("tools", []),
            )

        elif middleware_type == MiddlewareType.CONTEXT_EDITING:
            edit = ClearToolUsesEdit(
                trigger=merged_options.get("trigger", 10000),
                keep=merged_options.get("keep", 5),
                placeholder=merged_options.get("placeholder", "[cleared]"),
            )
            return ContextEditingMiddleware(edits=[edit])

        elif middleware_type == MiddlewareType.SHELL_TOOL:
            policy = merged_options.get("execution_policy", "host")
            execution_policy = HostExecutionPolicy() if policy == "host" else DockerExecutionPolicy()
            shell_kwargs = {
                "workspace_root": merged_options.get("workspace_root", "."),
                "execution_policy": execution_policy,
            }
            # Pass shell_command if specified (important for Windows compatibility)
            if merged_options.get("shell_command"):
                shell_kwargs["shell_command"] = merged_options["shell_command"]
            return ShellToolMiddleware(**shell_kwargs)

        elif middleware_type == MiddlewareType.FILE_SEARCH:
            return FilesystemFileSearchMiddleware(
                root_path=merged_options.get("root_path", "."),
                use_ripgrep=merged_options.get("use_ripgrep", True),
                max_file_size_mb=merged_options.get("max_file_size_mb", 10),
            )

        elif middleware_type == MiddlewareType.SECRET_INJECTION:
            return SecretInjectionMiddleware(
                secrets_manager=merged_options.get("secrets_manager"),
                secret_names=merged_options.get("secret_names", []),
            )

        raise ValueError(f"Middleware type not implemented: {middleware_type}")

    @staticmethod
    def create_middleware_list(
        middleware_types: List[MiddlewareType],
        model: Any = None,
        options: Optional[Dict[MiddlewareType, Dict[str, Any]]] = None,
    ) -> List[Any]:
        """
        Create a list of middleware instances for use with create_agent.

        Args:
            middleware_types: List of middleware types to create
            model: LLM model (used for middleware that require it)
            options: Optional per-middleware options

        Returns:
            List of LangChain middleware instances
        """
        options = options or {}
        middleware_list = []

        for mw_type in middleware_types:
            mw_options = options.get(mw_type)
            middleware = MiddlewareFactory.create_middleware(mw_type, model, mw_options)
            middleware_list.append(middleware)

        return middleware_list


class MiddlewareStack:
    """Helper class for managing middleware configuration."""

    def __init__(self):
        self._middleware_types: List[MiddlewareType] = []
        self._options: Dict[MiddlewareType, Dict[str, Any]] = {}

    def add_middleware(
        self,
        middleware_type: MiddlewareType,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Add middleware to the stack."""
        if middleware_type not in self._middleware_types:
            self._middleware_types.append(middleware_type)
        if options:
            self._options[middleware_type] = options

    def remove_middleware(self, middleware_type: MiddlewareType):
        """Remove middleware from the stack."""
        if middleware_type in self._middleware_types:
            self._middleware_types.remove(middleware_type)
        self._options.pop(middleware_type, None)

    def get_middleware_list(self, model: Any = None) -> List[Any]:
        """Get list of middleware instances for create_agent."""
        return MiddlewareFactory.create_middleware_list(
            self._middleware_types,
            model,
            self._options,
        )

    def get_all_configs(self) -> List[MiddlewareConfig]:
        """Get all available middleware configurations."""
        return MiddlewareFactory.get_all_middleware_configs()

    def get_enabled_types(self) -> List[MiddlewareType]:
        """Get list of enabled middleware types."""
        return self._middleware_types.copy()


# Middleware presets for common use cases
MIDDLEWARE_PRESETS: Dict[str, List[MiddlewareType]] = {
    "minimal": [
        MiddlewareType.CONTEXT_EDITING,
    ],
    "standard": [
        MiddlewareType.SUMMARIZATION,
        MiddlewareType.TODOLIST,
        MiddlewareType.TOOL_RETRY,
        MiddlewareType.MODEL_RETRY,
        MiddlewareType.CONTEXT_EDITING,
    ],
    "full": [
        MiddlewareType.SUMMARIZATION,
        MiddlewareType.HUMAN_IN_LOOP,
        MiddlewareType.MODEL_CALL_LIMIT,
        MiddlewareType.TOOL_CALL_LIMIT,
        MiddlewareType.MODEL_FALLBACK,
        MiddlewareType.PII_DETECTION,
        MiddlewareType.TODOLIST,
        MiddlewareType.LLM_TOOL_SELECTOR,
        MiddlewareType.TOOL_RETRY,
        MiddlewareType.MODEL_RETRY,
        MiddlewareType.CONTEXT_EDITING,
    ],
    "safe": [
        MiddlewareType.HUMAN_IN_LOOP,
        MiddlewareType.PII_DETECTION,
        MiddlewareType.MODEL_CALL_LIMIT,
        MiddlewareType.TOOL_CALL_LIMIT,
    ],
    "coding": [
        MiddlewareType.SUMMARIZATION,
        MiddlewareType.TODOLIST,
        MiddlewareType.SHELL_TOOL,
        MiddlewareType.FILE_SEARCH,
        MiddlewareType.TOOL_RETRY,
        MiddlewareType.MODEL_RETRY,
        MiddlewareType.CONTEXT_EDITING,
    ],
}


# Re-export built-in middleware classes for direct use
__all__ = [
    # Enums and configs
    "MiddlewareType",
    "MiddlewareConfig",
    "MIDDLEWARE_CONFIGS",
    "MIDDLEWARE_PRESETS",
    # Factory and stack
    "MiddlewareFactory",
    "MiddlewareStack",
    # Built-in LangChain middleware (re-exported)
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
]
