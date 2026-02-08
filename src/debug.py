"""
Debug and Logging Module for LangChain Agents

Provides logging utilities and execution phase tracking.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading


class DebugLevel(Enum):
    """Debug verbosity levels."""
    MINIMAL = 0
    NORMAL = 1
    VERBOSE = 2
    TRACE = 3


class AgentPhase(Enum):
    """Phases of agent execution."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    SENDING_REQUEST = "sending_request"
    WAITING_RESPONSE = "waiting_response"
    STREAMING = "streaming"
    PROCESSING_TOOLS = "processing_tools"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class DebugEvent:
    """Single debug event."""
    timestamp: datetime
    phase: AgentPhase
    message: str
    details: Optional[Dict[str, Any]] = None
    level: DebugLevel = DebugLevel.NORMAL
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase.value,
            "message": self.message,
            "details": self.details,
            "level": self.level.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    def format(self) -> str:
        """Format event for display."""
        ts = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        duration = f" ({self.duration_ms:.0f}ms)" if self.duration_ms else ""
        error = f" ERROR: {self.error}" if self.error else ""
        return f"[{ts}] [{self.phase.value.upper()}] {self.message}{duration}{error}"


class DebugLogger:
    """Debug logger with event streaming."""

    def __init__(
        self,
        level: DebugLevel = DebugLevel.NORMAL,
        max_events: int = 1000,
        callback: Optional[Callable[[DebugEvent], None]] = None
    ):
        self.level = level
        self.max_events = max_events
        self.callback = callback
        self.events: deque = deque(maxlen=max_events)
        self.current_phase = AgentPhase.IDLE
        self.phase_start_time: Optional[float] = None
        self._lock = threading.Lock()

        # Setup Python logger
        self.logger = logging.getLogger("langchain_agent.debug")
        self.logger.setLevel(logging.DEBUG)

    def log(
        self,
        phase: AgentPhase,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        level: DebugLevel = DebugLevel.NORMAL,
        error: Optional[str] = None
    ):
        """Log a debug event."""
        if level.value > self.level.value:
            return

        duration_ms = None
        if phase != self.current_phase and self.phase_start_time:
            duration_ms = (time.time() - self.phase_start_time) * 1000

        event = DebugEvent(
            timestamp=datetime.now(),
            phase=phase,
            message=message,
            details=details,
            level=level,
            duration_ms=duration_ms,
            error=error,
        )

        with self._lock:
            self.events.append(event)
            if phase != self.current_phase:
                self.current_phase = phase
                self.phase_start_time = time.time()

        # Python logging
        log_msg = event.format()
        if error:
            self.logger.error(log_msg)
        elif level == DebugLevel.TRACE:
            self.logger.debug(log_msg)
        else:
            self.logger.info(log_msg)

        if self.callback:
            try:
                self.callback(event)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    def error(self, message: str, error: Optional[Exception] = None):
        """Log error."""
        error_str = str(error) if error else None
        self.log(AgentPhase.ERROR, message, error=error_str)

    def get_events(self, limit: int = 100) -> List[DebugEvent]:
        """Get recent events."""
        with self._lock:
            return list(self.events)[-limit:]

    def clear(self):
        """Clear event history."""
        with self._lock:
            self.events.clear()
            self.current_phase = AgentPhase.IDLE
            self.phase_start_time = None


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get or create the global debug logger."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(level=DebugLevel.VERBOSE)
    return _debug_logger


def set_debug_level(level: DebugLevel):
    """Set the global debug level."""
    get_debug_logger().level = level
