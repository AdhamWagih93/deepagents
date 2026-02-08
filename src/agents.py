"""
LangChain Agent Factory

This module provides factory functions for creating LangChain agents with Ollama
using the `create_agent` function from langchain.agents.

Based on working implementation from sample_agent.py.
"""

from typing import Any, Dict, List, Optional, Union, Generator
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# LLM handling for Ollama
from langchain_ollama.chat_models import ChatOllama

# Message handling
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Tool handling
from langchain.tools import tool, BaseTool

# Agent creation
from langchain.agents import create_agent

# Checkpointer
from langgraph.checkpoint.memory import InMemorySaver

# Local imports
from .ollama_integration import OllamaConfig, create_ollama_llm
from .middleware import MiddlewareFactory, MiddlewareType, MiddlewareStack


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    name: str = "LangChain Agent"
    description: str = "A helpful AI assistant"
    system_prompt: Optional[str] = None
    model_config: Optional[OllamaConfig] = None
    middleware_types: List[MiddlewareType] = field(default_factory=list)
    middleware_options: Dict[MiddlewareType, Dict[str, Any]] = field(default_factory=dict)
    tools: List[Any] = field(default_factory=list)  # List of @tool decorated functions
    max_iterations: int = 25
    use_checkpointer: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "model": self.model_config.to_dict() if self.model_config else None,
            "middleware": [m.value for m in self.middleware_types],
            "tools": [getattr(t, 'name', str(t)) for t in self.tools] if self.tools else [],
            "max_iterations": self.max_iterations,
            "use_checkpointer": self.use_checkpointer,
        }


class AgentFactory:
    """Factory for creating LangChain agents with Ollama."""

    @staticmethod
    def get_agent_type_info() -> Dict[str, Dict[str, Any]]:
        """Get information about agent configuration options."""
        return {
            "default": {
                "name": "Default Agent",
                "description": "Standard LangChain agent with configurable tools and middleware.",
                "middleware": [],
                "tools": [],
                "use_cases": ["General purpose", "Customizable"],
            }
        }

    @staticmethod
    def create_agent(
        config: AgentConfig,
        checkpointer: Optional[Any] = None,
        context_schema: Optional[type] = None,
    ):
        """
        Create a LangChain agent using create_agent from langchain.agents.

        Args:
            config: Agent configuration
            checkpointer: Optional checkpointer for state persistence (defaults to InMemorySaver)
            context_schema: Optional dataclass for custom context

        Returns:
            Compiled LangChain agent
        """
        # Create the LLM using ChatOllama
        llm = create_ollama_llm(config.model_config)

        # Build system prompt
        system_prompt = config.system_prompt or f"""You are {config.name}, a helpful AI assistant.
{config.description}

Be concise, accurate, and helpful. When you need to perform actions, use the available tools.
Always explain your reasoning before taking actions."""

        # Create middleware list from types (pass model for middleware that need it)
        middleware = None
        if config.middleware_types:
            middleware = MiddlewareFactory.create_middleware_list(
                config.middleware_types,
                model=llm,
                options=config.middleware_options,
            )

        # Create checkpointer if not provided
        if config.use_checkpointer and checkpointer is None:
            checkpointer = InMemorySaver()

        # Create agent using LangChain's create_agent
        agent_kwargs = {
            "model": llm,
            "tools": config.tools or [],
            "system_prompt": system_prompt,
        }

        if middleware:
            agent_kwargs["middleware"] = middleware

        if checkpointer:
            agent_kwargs["checkpointer"] = checkpointer

        if context_schema:
            agent_kwargs["context_schema"] = context_schema

        agent = create_agent(**agent_kwargs)
        return agent


class LangChainAgent:
    """
    High-level wrapper for LangChain agents with streaming support.

    Based on sample_agent.py implementation.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        checkpointer: Optional[Any] = None,
        context_schema: Optional[type] = None,
    ):
        self.config = config or AgentConfig()
        self.context_schema = context_schema
        self.agent = AgentFactory.create_agent(self.config, checkpointer, context_schema)
        self.thread_id = str(uuid.uuid4())
        self.message_history: List[BaseMessage] = []
        self._last_result: Optional[Dict[str, Any]] = None

    def invoke(
        self,
        user_message: str,
        thread_id: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a user message.

        Args:
            user_message: The user's input message
            thread_id: Optional thread ID for conversation continuity
            context: Optional context object (must match context_schema)

        Returns:
            Agent response with messages and metadata
        """
        thread_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": thread_id}}

        invoke_kwargs = {
            "input": {"messages": [{"role": "user", "content": user_message}]},
            "config": config,
        }

        if context:
            invoke_kwargs["context"] = context

        result = self.agent.invoke(**invoke_kwargs)
        self._last_result = result
        return result

    def stream(
        self,
        user_message: str,
        thread_id: Optional[str] = None,
        context: Optional[Any] = None,
        stream_mode: str = "updates",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream agent response.

        Args:
            user_message: The user's input message
            thread_id: Optional thread ID for conversation continuity
            context: Optional context object (must match context_schema)
            stream_mode: "updates" for step-by-step, "values" for full state

        Yields:
            Streaming events from the agent with step and data
        """
        thread_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": thread_id}}

        stream_kwargs = {
            "input": {"messages": [{"role": "user", "content": user_message}]},
            "config": config,
            "stream_mode": stream_mode,
        }

        if context:
            stream_kwargs["context"] = context

        for chunk in self.agent.stream(**stream_kwargs):
            yield chunk

    def get_last_response(self) -> Optional[str]:
        """Get the last assistant response content."""
        if self._last_result and "messages" in self._last_result:
            messages = self._last_result["messages"]
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    return last_msg.content
        return None


@dataclass
class AgentInteraction:
    """Record of a single agent interaction."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: str = ""
    user_message: str = ""
    assistant_message: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # New fields for detailed tracing
    steps: List[Dict[str, Any]] = field(default_factory=list)
    middleware_events: List[Dict[str, Any]] = field(default_factory=list)
    human_approvals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "tool_calls": self.tool_calls,
            "todos": self.todos,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
            "metadata": self.metadata,
            "steps": self.steps,
            "middleware_events": self.middleware_events,
            "human_approvals": self.human_approvals,
        }


@dataclass
class AgentSession:
    """Session containing multiple agent interactions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    agent_config: Optional[AgentConfig] = None
    interactions: List[AgentInteraction] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_interaction(self, interaction: AgentInteraction):
        """Add an interaction to the session."""
        self.interactions.append(interaction)
        self.total_duration_ms += interaction.duration_ms
        self.total_tokens_in += interaction.tokens_in
        self.total_tokens_out += interaction.tokens_out

    def get_message_history(self) -> List[Dict[str, str]]:
        """Get message history for LangChain."""
        history = []
        for interaction in self.interactions:
            if interaction.user_message:
                history.append({"role": "user", "content": interaction.user_message})
            if interaction.assistant_message:
                history.append({"role": "assistant", "content": interaction.assistant_message})
        return history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "thread_id": self.thread_id,
            "agent_config": self.agent_config.to_dict() if self.agent_config else None,
            "interactions": [i.to_dict() for i in self.interactions],
            "stats": {
                "total_interactions": len(self.interactions),
                "total_duration_ms": self.total_duration_ms,
                "total_tokens_in": self.total_tokens_in,
                "total_tokens_out": self.total_tokens_out,
            }
        }
