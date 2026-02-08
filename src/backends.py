"""
Backend Configurations for LangChain Agents

This module provides backend options for state and memory management:
- MemorySaver: In-memory checkpointing for conversation state
- SQLite: Persistent storage for conversation history
"""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import os


class BackendType(Enum):
    """Available backend types."""
    MEMORY = "memory"
    SQLITE = "sqlite"


@dataclass
class BackendConfig:
    """Configuration for a backend instance."""
    backend_type: BackendType
    name: str
    description: str
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.backend_type.value,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "options": self.options
        }


class BackendFactory:
    """Factory for creating backend instances."""

    @staticmethod
    def get_all_backend_configs() -> List[BackendConfig]:
        """Return all available backend configurations."""
        return [
            BackendConfig(
                backend_type=BackendType.MEMORY,
                name="MemorySaver",
                description="In-memory checkpointing using LangGraph MemorySaver. State persists during the session but is lost when the application restarts.",
                options={}
            ),
            BackendConfig(
                backend_type=BackendType.SQLITE,
                name="SQLiteSaver",
                description="Persistent SQLite-based checkpointing. Conversation state is saved to disk and persists across restarts.",
                options={
                    "db_path": "langchain_checkpoints.db"
                }
            ),
        ]

    @staticmethod
    def create_checkpointer(config: BackendConfig):
        """
        Create a checkpointer instance from configuration.

        Args:
            config: Backend configuration

        Returns:
            Checkpointer instance
        """
        if config.backend_type == BackendType.MEMORY:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

        elif config.backend_type == BackendType.SQLITE:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                db_path = config.options.get("db_path", "langchain_checkpoints.db")
                return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
            except ImportError:
                # Fallback to memory saver
                from langgraph.checkpoint.memory import MemorySaver
                return MemorySaver()

        # Default to memory saver
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


@dataclass
class BackendMetrics:
    """Metrics tracked for backend operations."""
    checkpoint_saves: int = 0
    checkpoint_loads: int = 0
    total_bytes_saved: int = 0
    errors: List[str] = field(default_factory=list)

    def record_save(self, bytes_count: int = 0):
        """Record a checkpoint save."""
        self.checkpoint_saves += 1
        self.total_bytes_saved += bytes_count

    def record_load(self):
        """Record a checkpoint load."""
        self.checkpoint_loads += 1

    def record_error(self, error: str):
        """Record an error."""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_saves": self.checkpoint_saves,
            "checkpoint_loads": self.checkpoint_loads,
            "total_bytes_saved": self.total_bytes_saved,
            "error_count": len(self.errors)
        }
