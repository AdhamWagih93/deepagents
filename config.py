"""
Configuration Management for DeepAgents Application

Loads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class OllamaSettings:
    """Ollama connection settings."""
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q6_K"))
    timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "120")))


@dataclass
class StreamlitSettings:
    """Streamlit application settings."""
    port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501")))
    theme: str = field(default_factory=lambda: os.getenv("STREAMLIT_THEME", "light"))


@dataclass
class DatabaseSettings:
    """Database settings for history storage."""
    db_path: str = field(default_factory=lambda: os.getenv("HISTORY_DB_PATH", "deepagents_history.db"))


@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


@dataclass
class AppConfig:
    """Complete application configuration."""
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    streamlit: StreamlitSettings = field(default_factory=StreamlitSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment."""
        return cls()


# Global config instance
config = AppConfig.load()
