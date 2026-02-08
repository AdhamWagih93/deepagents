"""
LangChain Agent Explorer - Comprehensive Streamlit Application

A feature-rich chat interface for exploring LangChain agents with Ollama,
built-in middleware, and comprehensive tracing.

Features:
- LangChain agents with Ollama integration using create_agent
- 14 built-in middleware types from langchain.agents.middleware
- Real-time streaming with step-by-step visualization
- Human-in-the-loop approval for tool calls
- Tool call visualization with inputs/outputs
- Middleware event tracking
- Todo list display with real agent todos
- Non-blocking concurrent prompt execution
- Scheduled prompts system
- Historical interaction browser
- Enhanced logging and debug panel
"""

import streamlit as st
import time
import uuid
import json
import threading
import logging
import queue
import shutil
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DeepAgents")

# Page configuration - No sidebar, wide layout
st.set_page_config(
    page_title="LangChain Agent Explorer",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import project modules
from src.backends import BackendFactory, BackendType, BackendConfig
from src.middleware import (
    MiddlewareFactory, MiddlewareType, MiddlewareStack,
    MIDDLEWARE_PRESETS, MIDDLEWARE_CONFIGS,
    TodoListMiddleware, ShellToolMiddleware, FilesystemFileSearchMiddleware,
    HostExecutionPolicy
)
from src.agents import AgentFactory, AgentConfig, LangChainAgent, AgentInteraction, AgentSession
from src.ollama_integration import (
    OllamaConfig, OllamaClient, create_ollama_llm, OLLAMA_PRESETS,
    test_ollama_connection, test_model_availability, test_model_inference
)
from src.history import HistoryStore, InteractionTracker, ToolCallRecord, TodoRecord
from src.tools import get_tool_registry, TOOL_PRESETS

# LangChain imports
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# For connection diagnostics
import socket
import httpx


# ============================================================================
# Connection Diagnostics
# ============================================================================

def diagnose_connection_error(error: Exception, base_url: str) -> str:
    """Analyze a connection error and return a helpful diagnostic message."""
    error_str = str(error)
    error_type = type(error).__name__

    # Parse URL
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 11434
    except Exception:
        host = "localhost"
        port = 11434

    diagnostics = [f"**Connection Error:** {error_type}"]
    diagnostics.append(f"**Target:** {base_url}")

    # Check specific error types
    if "10061" in error_str or "Connection refused" in error_str.lower():
        diagnostics.append("")
        diagnostics.append("**Diagnosis:** Ollama is not running or not accepting connections.")
        diagnostics.append("")
        diagnostics.append("**Solutions:**")
        diagnostics.append("1. Start Ollama: `ollama serve` (in a terminal)")
        diagnostics.append("2. Or start Ollama Desktop application")
        diagnostics.append(f"3. Verify Ollama is listening on port {port}")

        # Quick port check
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result != 0:
                diagnostics.append(f"4. ⚠️ Port {port} is NOT open on {host}")
            else:
                diagnostics.append(f"4. ✅ Port {port} is open (service may be starting)")
        except Exception:
            pass

    elif "timeout" in error_str.lower():
        diagnostics.append("")
        diagnostics.append("**Diagnosis:** Connection timed out.")
        diagnostics.append("")
        diagnostics.append("**Solutions:**")
        diagnostics.append("1. Check if Ollama is responding: `curl http://localhost:11434/api/tags`")
        diagnostics.append("2. The model may be loading (can take several minutes)")
        diagnostics.append("3. Try clicking '🔥 Load Model' first")

    elif "name resolution" in error_str.lower() or "getaddrinfo" in error_str.lower():
        diagnostics.append("")
        diagnostics.append("**Diagnosis:** Cannot resolve hostname.")
        diagnostics.append("")
        diagnostics.append("**Solutions:**")
        diagnostics.append(f"1. Check that '{host}' is a valid hostname")
        diagnostics.append("2. Try using 'localhost' or '127.0.0.1' instead")

    else:
        diagnostics.append("")
        diagnostics.append("**Full error:**")
        diagnostics.append(f"```\n{error_str}\n```")

    return "\n".join(diagnostics)


# ============================================================================
# Custom Context Schema (matching sample_agent.py)
# ============================================================================

@dataclass
class UserContext:
    """Custom context for agent interactions."""
    project: str = "default"
    environment: str = "development"
    user_role: str = "user"


# ============================================================================
# Scheduled Prompt System
# ============================================================================

class ScheduleInterval(Enum):
    """Supported scheduling intervals."""
    SECONDS_10 = ("10 seconds", 10)
    SECONDS_30 = ("30 seconds", 30)
    MINUTE_1 = ("1 minute", 60)
    MINUTES_5 = ("5 minutes", 300)
    MINUTES_15 = ("15 minutes", 900)
    MINUTES_30 = ("30 minutes", 1800)
    HOUR_1 = ("1 hour", 3600)
    HOURS_6 = ("6 hours", 21600)
    HOURS_12 = ("12 hours", 43200)
    HOURS_24 = ("24 hours", 86400)

    def __init__(self, label: str, seconds: int):
        self.label = label
        self.seconds = seconds


@dataclass
class ScheduledPrompt:
    """A scheduled prompt configuration."""
    id: str
    prompt: str
    interval: ScheduleInterval
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def should_run(self) -> bool:
        """Check if this scheduled prompt should run now."""
        if not self.enabled:
            return False
        if self.next_run is None:
            return True
        return datetime.now() >= self.next_run

    def update_after_run(self, result: Dict[str, Any]):
        """Update state after a run."""
        self.last_run = datetime.now()
        self.next_run = self.last_run + timedelta(seconds=self.interval.seconds)
        self.run_count += 1
        self.results.append({
            "timestamp": self.last_run.isoformat(),
            "result": result.get("message", "")[:500],
            "duration_ms": result.get("duration_ms", 0),
            "success": "error" not in result
        })
        # Keep only last 20 results
        if len(self.results) > 20:
            self.results = self.results[-20:]


# ============================================================================
# Non-blocking Execution Manager
# ============================================================================

class ExecutionManager:
    """Manages concurrent, non-blocking prompt execution."""

    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Future] = {}
        self.active_task_info: Dict[str, Dict[str, Any]] = {}  # Live task info
        self.results_queue: queue.Queue = queue.Queue()
        self.task_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        logger.info(f"ExecutionManager initialized with {max_workers} workers")

    def submit_prompt(
        self,
        task_id: str,
        prompt: str,
        agent,
        config: Dict,
        context: Any,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a prompt for non-blocking execution."""
        logger.info(f"Submitting task {task_id}: {prompt[:50]}...")

        def execute():
            start_time = time.time()
            result = {
                "task_id": task_id,
                "prompt": prompt,
                "started_at": datetime.now().isoformat(),
                "status": "running"
            }

            # Initialize live task info
            with self._lock:
                self.active_task_info[task_id] = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "started_at": datetime.now().isoformat(),
                    "status": "running",
                    "current_step": "initializing",
                    "steps": [],
                    "tool_calls": [],
                    "partial_response": "",
                    "elapsed_ms": 0,
                }

            try:
                # Collect streaming response
                full_response = []
                tool_calls = []
                todos = []
                steps = []

                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                    context=context,
                    stream_mode="updates",
                ):
                    for step_name, data in chunk.items():
                        step_info = {
                            "step": step_name,
                            "timestamp": datetime.now().isoformat()
                        }

                        # Extract content from messages
                        if "messages" in data:
                            for msg in data["messages"]:
                                if hasattr(msg, 'content'):
                                    content = msg.content
                                    if content:
                                        full_response.append(str(content))
                                        step_info["content"] = str(content)[:200]

                                # Extract tool calls
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_call = {
                                            "name": tc.get("name", "unknown"),
                                            "args": tc.get("args", {}),
                                            "id": tc.get("id", str(uuid.uuid4()))
                                        }
                                        tool_calls.append(tool_call)
                                        step_info["tool_call"] = tool_call

                                # Extract todos from additional_kwargs
                                if hasattr(msg, 'additional_kwargs'):
                                    kwargs = msg.additional_kwargs
                                    if 'todos' in kwargs:
                                        for todo in kwargs['todos']:
                                            todos.append(todo)

                        steps.append(step_info)
                        logger.debug(f"Task {task_id} step: {step_name}")

                        # Update live task info
                        with self._lock:
                            if task_id in self.active_task_info:
                                self.active_task_info[task_id].update({
                                    "current_step": step_name,
                                    "steps": steps.copy(),
                                    "tool_calls": tool_calls.copy(),
                                    "partial_response": "".join(full_response)[:500],
                                    "elapsed_ms": (time.time() - start_time) * 1000,
                                })

                duration = (time.time() - start_time) * 1000
                result.update({
                    "status": "completed",
                    "message": "".join(full_response),
                    "duration_ms": duration,
                    "tool_calls": tool_calls,
                    "todos": todos,
                    "steps": steps,
                    "completed_at": datetime.now().isoformat()
                })
                logger.info(f"Task {task_id} completed in {duration:.0f}ms")

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                error_msg = str(e)
                result.update({
                    "status": "error",
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "duration_ms": duration,
                    "completed_at": datetime.now().isoformat()
                })
                logger.error(f"Task {task_id} failed: {error_msg}")

            # Clean up live task info
            with self._lock:
                if task_id in self.active_task_info:
                    del self.active_task_info[task_id]

            # Put result in queue and call callback
            self.results_queue.put(result)
            if callback:
                try:
                    callback(result)
                except Exception as cb_error:
                    logger.warning(f"Callback error for task {task_id}: {cb_error}")

            with self._lock:
                self.task_history.append(result)
                if len(self.task_history) > 100:
                    self.task_history = self.task_history[-100:]

            return result

        future = self.executor.submit(execute)
        with self._lock:
            self.active_tasks[task_id] = future

        return task_id

    def get_active_count(self) -> int:
        """Get number of active tasks."""
        with self._lock:
            return sum(1 for f in self.active_tasks.values() if not f.done())

    def get_active_tasks_info(self) -> List[Dict[str, Any]]:
        """Get live info for all active tasks."""
        with self._lock:
            return list(self.active_task_info.values())

    def get_completed_results(self) -> List[Dict[str, Any]]:
        """Get all completed results from queue (non-blocking)."""
        results = []
        while not self.results_queue.empty():
            try:
                results.append(self.results_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get task execution history."""
        with self._lock:
            return self.task_history.copy()

    def cancel_task(self, task_id: str) -> bool:
        """Attempt to cancel a task."""
        with self._lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].cancel()
        return False

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)
        logger.info("ExecutionManager shutdown")


# ============================================================================
# Cached Resources
# ============================================================================

# Version hash to invalidate cache when code changes - increment when classes change
_CACHE_VERSION = "v3"

@st.cache_resource
def get_history_store(_version: str = _CACHE_VERSION):
    """Get cached history store instance."""
    logger.info(f"Initializing HistoryStore ({_version})")
    return HistoryStore()


@st.cache_resource
def get_interaction_tracker(_version: str = _CACHE_VERSION):
    """Get cached interaction tracker."""
    logger.info(f"Initializing InteractionTracker ({_version})")
    return InteractionTracker(get_history_store(_version))


@st.cache_resource
def get_execution_manager(_version: str = _CACHE_VERSION):
    """Get cached execution manager for non-blocking execution."""
    logger.info(f"Initializing ExecutionManager ({_version})")
    return ExecutionManager(max_workers=3)


@st.cache_resource
def get_checkpointer():
    """Get cached checkpointer for agent state."""
    logger.info("Initializing InMemorySaver checkpointer")
    return InMemorySaver()


@st.cache_data(ttl=60)
def get_backend_configs():
    """Cache backend configurations."""
    return BackendFactory.get_all_backend_configs()


@st.cache_data(ttl=60)
def get_middleware_configs():
    """Cache middleware configurations."""
    return MiddlewareFactory.get_all_middleware_configs()


# ============================================================================
# CSS Styling
# ============================================================================

@st.cache_data
def get_custom_css():
    return """
<style>
    /* ===== CSS VARIABLES ===== */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --surface: #ffffff;
        --surface-alt: #f8fafc;
        --border: #e2e8f0;
        --border-light: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
    }

    /* ===== BASE LAYOUT ===== */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ===== HIDE SIDEBAR COMPLETELY ===== */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* ===== TOP BAR STYLING ===== */
    .top-bar {
        background: white;
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        box-shadow: var(--shadow-sm);
    }
    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .top-bar-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* ===== PANEL CARDS ===== */
    .panel-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1rem;
        height: 100%;
    }
    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-light);
    }
    .panel-header h4 {
        margin: 0;
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* ===== AGENT CARD IN BUILDER ===== */
    .agent-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.15s ease;
    }
    .agent-card:hover {
        border-color: var(--primary-light);
        box-shadow: var(--shadow-sm);
    }
    .agent-card.active {
        border-color: var(--success);
        background: linear-gradient(135deg, #ecfdf5 0%, white 100%);
    }

    /* ===== MINI TABS FOR INLINE PANELS ===== */
    .mini-tabs {
        display: flex;
        gap: 0.25rem;
        background: var(--surface-alt);
        padding: 0.25rem;
        border-radius: var(--radius-md);
        margin-bottom: 0.75rem;
    }
    .mini-tab {
        padding: 0.4rem 0.75rem;
        border-radius: var(--radius-sm);
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.15s ease;
        border: none;
        background: transparent;
        color: var(--text-secondary);
    }
    .mini-tab:hover {
        background: rgba(99, 102, 241, 0.1);
    }
    .mini-tab.active {
        background: white;
        color: var(--primary);
        box-shadow: var(--shadow-sm);
    }

    /* ===== HEADER STYLING ===== */
    .app-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: var(--radius-lg);
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25), 0 8px 32px rgba(139, 92, 246, 0.15);
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .app-header p {
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
        position: relative;
        z-index: 1;
    }

    /* ===== STATUS BAR ===== */
    .status-bar {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 0.875rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1.75rem;
        flex-wrap: wrap;
        box-shadow: var(--shadow-sm);
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    .status-item strong {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* ===== DEBUG PANEL ===== */
    .debug-panel {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        color: #e2e8f0;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.75rem;
        padding: 1rem 1.25rem;
        border-radius: var(--radius-md);
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #334155;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    .debug-event {
        padding: 0.35rem 0.65rem;
        margin: 0.2rem 0;
        border-radius: var(--radius-sm);
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #475569;
        transition: all 0.15s ease;
    }
    .debug-event:hover {
        background: rgba(255,255,255,0.08);
    }
    .debug-event.error { border-left-color: #f87171; color: #fca5a5; background: rgba(239,68,68,0.12); }
    .debug-event.streaming { border-left-color: #34d399; color: #86efac; }
    .debug-event.tool { border-left-color: #60a5fa; color: #93c5fd; }
    .debug-event.middleware { border-left-color: #fbbf24; color: #fcd34d; }
    .debug-event.scheduled { border-left-color: #c084fc; color: #d8b4fe; }

    /* ===== STEP CARDS ===== */
    .step-card {
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.125rem 1.375rem;
        margin: 0.875rem 0;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .step-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    .step-card.thinking {
        border-left: 4px solid #a855f7;
        background: linear-gradient(135deg, #faf5ff 0%, var(--surface) 100%);
    }
    .step-card.tool {
        border-left: 4px solid #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, var(--surface) 100%);
    }
    .step-card.response {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, var(--surface) 100%);
    }
    .step-card.approval {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }

    /* ===== TOOL CALLS ===== */
    .tool-call {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: var(--radius-md);
        padding: 0.875rem 1.125rem;
        margin: 0.5rem 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
    }
    .tool-result {
        background: linear-gradient(135deg, #d1fae5 0%, #ecfdf5 100%);
        border: 1px solid #a7f3d0;
        border-radius: var(--radius-md);
        padding: 0.875rem 1.125rem;
        margin: 0.5rem 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
    }

    /* ===== STATUS BADGES ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.7rem;
        border-radius: var(--radius-xl);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        transition: all 0.15s ease;
    }
    .status-badge.healthy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 2px 6px rgba(16,185,129,0.35);
    }
    .status-badge.error {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
        box-shadow: 0 2px 6px rgba(239,68,68,0.35);
    }
    .status-badge.pending {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        box-shadow: 0 2px 6px rgba(245,158,11,0.35);
    }
    .status-badge.running {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        color: white;
        box-shadow: 0 2px 6px rgba(59,130,246,0.35);
    }

    /* ===== TODO LIST ===== */
    .todo-item {
        padding: 0.75rem 1.125rem;
        margin: 0.5rem 0;
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        gap: 0.625rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .todo-item:hover { transform: translateX(6px); }
    .todo-item.pending {
        background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
        border: 1px solid #fde68a;
    }
    .todo-item.in_progress {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border: 1px solid #93c5fd;
    }
    .todo-item.completed {
        background: linear-gradient(135deg, #d1fae5 0%, #ecfdf5 100%);
        border: 1px solid #6ee7b7;
        text-decoration: line-through;
        opacity: 0.75;
    }

    /* ===== TASK CARDS ===== */
    .task-card {
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.125rem 1.375rem;
        margin: 0.875rem 0;
        background: var(--surface);
        box-shadow: var(--shadow-md);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .task-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    .task-card.running {
        border-left: 4px solid #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, var(--surface) 100%);
    }
    .task-card.completed {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, var(--surface) 100%);
    }
    .task-card.error {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, var(--surface) 100%);
    }

    /* ===== SCHEDULE CARDS ===== */
    .schedule-card {
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.125rem;
        margin: 0.875rem 0;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    .schedule-card:hover {
        box-shadow: var(--shadow-md);
    }
    .schedule-card.enabled {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, var(--surface) 100%);
    }
    .schedule-card.disabled {
        border-left: 4px solid #94a3b8;
        opacity: 0.65;
    }

    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    .metric-card .value {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .label {
        font-size: 0.825rem;
        color: var(--text-muted);
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* ===== CHAT STYLING ===== */
    .stChatMessage {
        border-radius: var(--radius-lg) !important;
        margin-bottom: 0.875rem !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stChatMessageContent"] {
        padding: 1rem 1.25rem !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ===== PRIMARY BUTTONS ===== */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.375rem;
        background: var(--surface-alt);
        padding: 0.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-md) !important;
        padding: 0.625rem 1.125rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.08) !important;
        color: var(--primary) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--primary) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        border-radius: var(--radius-md) !important;
        background: var(--surface-alt) !important;
    }
    details[open] .streamlit-expanderHeader {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
    }

    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        transition: all 0.15s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }

    /* ===== DIVIDERS ===== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
        margin: 1.75rem 0 !important;
    }

    /* ===== LIVE TASK INDICATOR ===== */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.875rem;
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
        border-radius: var(--radius-xl);
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.35);
        animation: pulse-glow 2s infinite;
    }
    @keyframes pulse-glow {
        0%, 100% {
            opacity: 1;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.35);
        }
        50% {
            opacity: 0.85;
            box-shadow: 0 2px 16px rgba(239, 68, 68, 0.5);
        }
    }

    /* ===== CONNECTION STATUS ===== */
    .connection-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.875rem;
        border-radius: var(--radius-md);
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .connection-status.connected {
        background: linear-gradient(135deg, #d1fae5 0%, #ecfdf5 100%);
        color: #047857;
        border: 1px solid #6ee7b7;
    }
    .connection-status.disconnected {
        background: linear-gradient(135deg, #fee2e2 0%, #fef2f2 100%);
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .connection-status.unknown {
        background: var(--surface-alt);
        color: var(--text-secondary);
        border: 1px solid var(--border);
    }

    /* ===== CONVERSATION PANEL ===== */
    .conversation-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 0.875rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .conversation-card:hover {
        border-color: var(--primary-light);
        box-shadow: var(--shadow-sm);
    }
    .conversation-card.active {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-color: var(--primary);
    }

    /* ===== SCROLLBAR STYLING ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--surface-alt);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* ===== SELECTION ===== */
    ::selection {
        background: rgba(99, 102, 241, 0.2);
        color: var(--text-primary);
    }

    /* ===== SMOOTH TRANSITIONS FOR ALL ===== */
    * {
        scroll-behavior: smooth;
    }
</style>
"""


st.markdown(get_custom_css(), unsafe_allow_html=True)


# ============================================================================
# Session State
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Core state
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "thread_id": str(uuid.uuid4()),
        "ollama_config": OllamaConfig(),
        "agent_config": AgentConfig(),
        "agent_instance": None,

        # ===== Conversations System =====
        "conversations": {},  # Dict of conversation threads {thread_id: {name, messages, agent_id, created_at, updated_at}}
        "active_conversation_id": None,  # Currently active conversation thread

        # ===== Saved Agents System =====
        "saved_agents": {},  # Dict of saved agent configurations
        "active_agent_id": None,  # Currently selected agent for chat
        "agent_test_results": {},  # Test results for each agent
        "builder_agent_config": {  # Config being edited in builder
            "name": "New Agent",
            "description": "A helpful AI assistant",
            "system_prompt": "",
            "middleware": [MiddlewareType.TODOLIST, MiddlewareType.SHELL_TOOL, MiddlewareType.FILE_SEARCH],
            "tools": ["read_file", "write_file", "list_directory", "glob_search", "grep_search"],
            "model": "llama3.2:latest",
            "temperature": 0.7,
            "num_ctx": 8192,
        },

        # Middleware and tools (for active agent)
        "selected_middleware": [MiddlewareType.TODOLIST, MiddlewareType.SHELL_TOOL, MiddlewareType.FILE_SEARCH],
        "middleware_options": {},
        "selected_backend_type": BackendType.MEMORY,
        "selected_tools": ["read_file", "write_file", "list_directory", "glob_search", "grep_search"],

        # Connection state
        "ollama_connected": None,
        "available_models": [],
        "model_loaded": False,

        # Debug and tracing
        "debug_events": [],
        "debug_enabled": True,
        "current_phase": "idle",
        "is_streaming": False,

        # Execution tracing
        "current_steps": [],
        "tool_calls": [],
        "todos": [],
        "middleware_events": [],

        # Human-in-the-loop
        "pending_approvals": [],
        "approval_history": [],

        # Context
        "user_context": UserContext(),

        # Non-blocking execution
        "concurrent_tasks": {},
        "task_results": [],

        # Scheduled prompts
        "scheduled_prompts": {},
        "scheduler_running": False,
        "scheduler_last_check": None,

        # Agent presets (legacy, kept for compatibility)
        "agent_presets": {},
        "editing_preset": None,

        # Detailed debug info for last agent call
        "last_agent_call_info": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_debug_event(phase: str, message: str, event_type: str = "info", error: Optional[str] = None):
    """Add a debug event to session state."""
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "phase": phase,
        "message": message,
        "type": event_type,
        "error": error,
    }
    if "debug_events" not in st.session_state:
        st.session_state.debug_events = []
    st.session_state.debug_events.append(event)
    if len(st.session_state.debug_events) > 200:
        st.session_state.debug_events = st.session_state.debug_events[-200:]

    # Also log to Python logger
    log_level = logging.ERROR if event_type == "error" else logging.INFO
    logger.log(log_level, f"[{phase}] {message}")


def add_step(step_type: str, content: Any, metadata: Optional[Dict] = None):
    """Add an execution step for visualization."""
    step = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "type": step_type,  # thinking, tool, response, approval
        "content": content,
        "metadata": metadata or {},
    }
    st.session_state.current_steps.append(step)


def add_tool_call(tool_name: str, args: Dict, result: Any = None, status: str = "pending"):
    """Record a tool call."""
    call = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "args": args,
        "result": result,
        "status": status,
    }
    st.session_state.tool_calls.append(call)
    add_debug_event("TOOL", f"{tool_name}({json.dumps(args)[:50]}...)", "tool")
    return call["id"]


def update_tool_result(call_id: str, result: Any, status: str = "completed"):
    """Update a tool call with its result."""
    for call in st.session_state.tool_calls:
        if call["id"] == call_id:
            call["result"] = result
            call["status"] = status
            break


def add_todo(content: str, status: str = "pending"):
    """Add a todo item."""
    todo = {
        "id": str(uuid.uuid4()),
        "content": content,
        "status": status,
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.todos.append(todo)
    add_debug_event("TODO", f"[{status}] {content[:50]}...", "middleware")


def update_todos_from_response(todos_data: List[Dict]):
    """Update todos from agent response."""
    st.session_state.todos = []
    for todo in todos_data:
        add_todo(
            content=todo.get("content", todo.get("task", "Unknown")),
            status=todo.get("status", "pending")
        )


# ============================================================================
# Agent Creation
# ============================================================================

def sync_agent_config_from_active():
    """Sync session state configuration from the active agent.

    This ensures that when creating an agent, we use the correct middleware
    and tools from the currently active agent, not stale values.
    """
    active_id = st.session_state.active_agent_id
    if not active_id or active_id not in st.session_state.saved_agents:
        return  # No active agent to sync from

    agent_data = st.session_state.saved_agents[active_id]

    # Sync middleware - convert string values to MiddlewareType
    agent_middleware = agent_data.get("middleware") or []
    st.session_state.selected_middleware = [
        MiddlewareType(m) for m in agent_middleware
        if m in [mt.value for mt in MiddlewareType]
    ]

    # Sync tools
    st.session_state.selected_tools = agent_data.get("tools") or []

    # Sync agent config
    st.session_state.agent_config.name = agent_data.get("name", "Agent")
    st.session_state.agent_config.description = agent_data.get("description", "")
    st.session_state.agent_config.system_prompt = agent_data.get("system_prompt")

    # Sync Ollama config
    st.session_state.ollama_config.model = agent_data.get("model", "llama3.2:latest")
    st.session_state.ollama_config.temperature = agent_data.get("temperature", 0.7)
    st.session_state.ollama_config.num_ctx = agent_data.get("num_ctx", 8192)

    logger.info(f"Synced config from active agent: {agent_data.get('name')} with {len(st.session_state.selected_middleware)} middleware")


def create_configured_agent():
    """Create an agent with current configuration."""
    # IMPORTANT: Sync from active agent first to ensure correct middleware/tools
    sync_agent_config_from_active()

    config = st.session_state.ollama_config
    agent_cfg = st.session_state.agent_config

    add_debug_event("AGENT", "Creating agent with middleware...", "info")
    logger.info(f"Creating agent with model: {config.model}")

    # Create LLM
    llm = create_ollama_llm(config)

    # Build middleware list
    middleware = []
    middleware_names = []

    for mw_type in st.session_state.selected_middleware:
        try:
            mw_options = st.session_state.middleware_options.get(mw_type, {})

            # Special handling for shell tool - set workspace and shell command
            if mw_type == MiddlewareType.SHELL_TOOL:
                mw_options.setdefault("workspace_root", ".")
                mw_options.setdefault("execution_policy", "host")
                # Use appropriate shell for the platform
                if sys.platform == "win32":
                    mw_options.setdefault("shell_command", ["cmd.exe", "/Q", "/K"])
                else:
                    mw_options.setdefault("shell_command", ["/bin/bash"])

            # Special handling for file search
            if mw_type == MiddlewareType.FILE_SEARCH:
                mw_options.setdefault("root_path", ".")
                # Check if ripgrep is available, disable if not found
                ripgrep_available = shutil.which("rg") is not None
                mw_options.setdefault("use_ripgrep", ripgrep_available)
                if not ripgrep_available:
                    logger.info("ripgrep (rg) not found, using Python-based file search")

            mw = MiddlewareFactory.create_middleware(mw_type, model=llm, options=mw_options)
            middleware.append(mw)
            middleware_names.append(mw_type.value)
            logger.info(f"Added middleware: {mw_type.value}")
        except Exception as e:
            logger.warning(f"Failed to create middleware {mw_type}: {e}")
            add_debug_event("MIDDLEWARE", f"Failed: {mw_type.value} - {e}", "error")

    # Get tools
    tool_registry = get_tool_registry()
    tools = tool_registry.get_tools(st.session_state.selected_tools)
    logger.info(f"Tools configured: {st.session_state.selected_tools}")

    # System prompt
    system_prompt = agent_cfg.system_prompt or f"""You are {agent_cfg.name}, a helpful AI assistant.
{agent_cfg.description}

You have access to tools for file operations, search, and shell commands.
Use the todo list to track complex tasks. Be concise and helpful."""

    # Capture detailed agent call info for debugging
    agent_call_info = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
        "agent": {
            "name": agent_cfg.name,
            "description": agent_cfg.description,
        },
        "system_prompt": system_prompt,
        "middleware": middleware_names,
        "tools": st.session_state.selected_tools,
        "context_schema": "UserContext",
        "checkpointer": st.session_state.selected_backend_type.value,
    }
    st.session_state.last_agent_call_info = agent_call_info

    # Create agent
    try:
        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=middleware if middleware else [],  # Pass empty list, not None
            checkpointer=get_checkpointer(),
            context_schema=UserContext,
            system_prompt=system_prompt,
        )

        add_debug_event("AGENT", f"Agent created with {len(middleware)} middleware, {len(tools)} tools", "info")
        logger.info(f"Agent created successfully: {len(middleware)} middleware, {len(tools)} tools")

        return agent

    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        add_debug_event("AGENT", f"Creation failed: {e}", "error")
        raise


def get_or_create_agent():
    """Get existing agent or create a new one."""
    if st.session_state.agent_instance is None:
        st.session_state.agent_instance = create_configured_agent()
    return st.session_state.agent_instance


# ============================================================================
# UI Helpers
# ============================================================================

def render_agent_session_header():
    """Render a header showing current agent and session info."""
    active_id = st.session_state.active_agent_id
    if active_id and active_id in st.session_state.saved_agents:
        agent = st.session_state.saved_agents[active_id]
        agent_name = agent.get("name", "Unknown")
        agent_model = agent.get("model", "Unknown")
    else:
        agent_name = st.session_state.agent_config.name
        agent_model = st.session_state.ollama_config.model

    session_id = st.session_state.session_id[:8]
    thread_id = st.session_state.thread_id[:8]

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 1rem;
                margin-bottom: 1rem; display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.25rem;">🤖</span>
            <div>
                <strong style="color: #1e293b;">{agent_name}</strong>
                <br><span style="font-size: 0.75rem; color: #64748b;">{agent_model}</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1rem;">📋</span>
            <div>
                <span style="font-size: 0.8rem; color: #64748b;">Session</span>
                <br><code style="font-size: 0.75rem; color: #475569;">{session_id}</code>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1rem;">🧵</span>
            <div>
                <span style="font-size: 0.8rem; color: #64748b;">Thread</span>
                <br><code style="font-size: 0.75rem; color: #475569;">{thread_id}</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# UI Fragments
# ============================================================================

@st.fragment
def connection_status_fragment():
    """Fragment for connection status."""
    status = st.session_state.ollama_connected

    # Status display with visual indicator
    if status is None:
        st.markdown("""
        <div class="connection-status unknown">
            ⚪ Not Checked
        </div>
        """, unsafe_allow_html=True)
    elif status:
        st.markdown("""
        <div class="connection-status connected">
            🟢 Connected to Ollama
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="connection-status disconnected">
            🔴 Disconnected
        </div>
        """, unsafe_allow_html=True)

    def check_connection():
        """Callback to check Ollama connection."""
        try:
            result = test_ollama_connection(
                st.session_state.ollama_config.base_url,
                timeout=3.0
            )
            st.session_state.ollama_connected = result.success
            if result.success:
                client = OllamaClient(st.session_state.ollama_config)
                st.session_state.available_models = client.list_models()
                add_debug_event("CONNECTED", f"Connected ({result.latency_ms:.0f}ms)")
                st.toast(f"✅ Connected ({result.latency_ms:.0f}ms)", icon="🟢")
            else:
                add_debug_event("ERROR", f"Connection failed: {result.error}", "error")
                st.toast("❌ Connection failed", icon="🔴")
        except Exception as e:
            st.session_state.ollama_connected = False
            add_debug_event("ERROR", str(e), "error")
            st.toast("❌ Connection error", icon="🔴")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("🔄 Check", key="refresh_conn", help="Check connection", on_click=check_connection, use_container_width=True)

    with col2:
        # Model warmup button
        if st.session_state.ollama_connected:
            if st.session_state.get("model_loaded", False):
                st.button("✅ Ready", key="model_ready", disabled=True, use_container_width=True)
            else:
                if st.button("🔥 Load", key="warmup_btn", use_container_width=True):
                    with st.spinner("Loading model into memory..."):
                        client = OllamaClient(st.session_state.ollama_config)
                        success = client.warmup_model()
                        if success:
                            st.session_state.model_loaded = True
                            st.session_state.agent_instance = None
                            st.toast("✅ Model loaded successfully!", icon="🔥")
                            st.rerun()
                        else:
                            st.toast("❌ Failed to load model", icon="⚠️")
        else:
            st.button("🔥 Load", key="warmup_btn_disabled", disabled=True, use_container_width=True)


@st.fragment
def debug_panel_fragment():
    """Fragment for debug panel with detailed tracing."""
    # Phase indicator with enhanced styling
    phase = st.session_state.current_phase
    phase_config = {
        "idle": ("🔵", "#3b82f6", "Idle"),
        "sending": ("🟡", "#f59e0b", "Sending"),
        "streaming": ("🟢", "#22c55e", "Streaming"),
        "tool_call": ("🔧", "#a855f7", "Tool Call"),
        "waiting_approval": ("⏳", "#f97316", "Awaiting Approval"),
        "completed": ("✅", "#22c55e", "Completed"),
        "error": ("🔴", "#ef4444", "Error")
    }
    icon, color, label = phase_config.get(phase, ("⚪", "#94a3b8", phase))

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.25rem;">{icon}</span>
            <span style="font-weight: 600; color: {color};">{label.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        exec_mgr = get_execution_manager()
        active = exec_mgr.get_active_count()
        if active > 0:
            st.markdown(f"🔄 **{active} running**")

    with col3:
        if st.button("🗑️ Clear", key="clear_debug"):
            st.session_state.debug_events = []

    if st.session_state.debug_events:
        events_html = '<div class="debug-panel">'
        for event in reversed(st.session_state.debug_events[-30:]):
            cls = f" {event.get('type', 'info')}" if event.get("type") else ""
            if event.get("error"):
                cls += " error"
            events_html += f'<div class="debug-event{cls}">[{event["timestamp"]}] [{event["phase"]}] {event["message"]}</div>'
        events_html += '</div>'
        st.markdown(events_html, unsafe_allow_html=True)
    else:
        st.caption("No debug events yet")


@st.fragment
def step_visualization_fragment():
    """Fragment for visualizing execution steps."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">📋 Execution Steps</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Step-by-step visualization of agent execution flow
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    if not st.session_state.current_steps:
        st.info("No steps recorded yet. Start a conversation to see execution steps.")
        return

    for step in st.session_state.current_steps:
        step_type = step["type"]
        content = step["content"]

        if step_type == "thinking":
            st.markdown(f"""
            <div class="step-card thinking">
                <strong>🧠 Thinking</strong><br>
                <small>{step['timestamp']}</small><br>
                {content[:200]}{'...' if len(str(content)) > 200 else ''}
            </div>
            """, unsafe_allow_html=True)

        elif step_type == "tool":
            tool_name = step["metadata"].get("tool_name", "unknown")
            st.markdown(f"""
            <div class="step-card tool">
                <strong>🔧 Tool Call: {tool_name}</strong><br>
                <small>{step['timestamp']}</small>
                <div class="tool-call">Args: {json.dumps(step['metadata'].get('args', {}), indent=2)[:300]}</div>
                <div class="tool-result">Result: {str(content)[:300]}</div>
            </div>
            """, unsafe_allow_html=True)

        elif step_type == "response":
            st.markdown(f"""
            <div class="step-card response">
                <strong>💬 Response</strong><br>
                <small>{step['timestamp']}</small><br>
                {content[:500]}{'...' if len(str(content)) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)

        elif step_type == "approval":
            st.markdown(f"""
            <div class="step-card approval">
                <strong>⏳ Awaiting Approval</strong><br>
                <small>{step['timestamp']}</small><br>
                Tool: {step['metadata'].get('tool_name', 'unknown')}<br>
                Status: {step['metadata'].get('status', 'pending')}
            </div>
            """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Steps", key="clear_steps"):
        st.session_state.current_steps = []
        st.rerun()


@st.fragment
def tool_calls_fragment():
    """Fragment for displaying tool calls."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">🔧 Tool Calls</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            View tool invocations, arguments, and results
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    if not st.session_state.tool_calls:
        st.info("No tool calls yet. Tool invocations will appear here during agent interactions.")
        return

    for call in reversed(st.session_state.tool_calls[-10:]):
        status_icon = {"pending": "⏳", "completed": "✅", "failed": "❌", "approved": "👍", "rejected": "👎"}.get(call["status"], "⚪")

        with st.expander(f"{status_icon} **{call['tool_name']}** - {call['timestamp'][:19]}"):
            st.markdown("**Arguments:**")
            st.json(call["args"])

            if call["result"]:
                st.markdown("**Result:**")
                if isinstance(call["result"], (dict, list)):
                    st.json(call["result"])
                else:
                    st.code(str(call["result"])[:1000])


@st.fragment
def todo_list_fragment():
    """Fragment for displaying agent's todo list."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">📝 Agent Todo List</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Track agent-managed tasks and their completion status
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    # Check for TodoListMiddleware
    selected_mw = st.session_state.selected_middleware or []
    has_todo_middleware = MiddlewareType.TODOLIST in selected_mw

    if not has_todo_middleware:
        st.warning("⚠️ TodoListMiddleware not enabled. Enable it in Agent Builder to use task tracking.")

    if not st.session_state.todos:
        if has_todo_middleware:
            st.info("No todos yet. The agent will create todos when planning complex tasks.")
        return

    # Group by status
    pending = [t for t in st.session_state.todos if t.get("status") == "pending"]
    in_progress = [t for t in st.session_state.todos if t.get("status") == "in_progress"]
    completed = [t for t in st.session_state.todos if t.get("status") == "completed"]

    if in_progress:
        st.markdown("**🔄 In Progress:**")
        for todo in in_progress:
            st.markdown(f"""
            <div class="todo-item in_progress">
                🔄 {todo.get('content', 'Unknown task')}
            </div>
            """, unsafe_allow_html=True)

    if pending:
        st.markdown("**⬜ Pending:**")
        for todo in pending:
            st.markdown(f"""
            <div class="todo-item pending">
                ⬜ {todo.get('content', 'Unknown task')}
            </div>
            """, unsafe_allow_html=True)

    if completed:
        st.markdown("**✅ Completed:**")
        for todo in completed:
            st.markdown(f"""
            <div class="todo-item completed">
                ✅ {todo.get('content', 'Unknown task')}
            </div>
            """, unsafe_allow_html=True)

    # Summary
    st.caption(f"Total: {len(pending)} pending, {len(in_progress)} in progress, {len(completed)} completed")


@st.fragment
def human_approval_fragment():
    """Fragment for human-in-the-loop approval."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">👤 Human Approval Queue</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Review and approve tool calls before execution
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    pending = st.session_state.pending_approvals
    selected_mw = st.session_state.selected_middleware or []
    if not pending:
        has_hitl = MiddlewareType.HUMAN_IN_LOOP in selected_mw
        if has_hitl:
            st.info("No pending approvals.")
        else:
            st.info("Enable HumanInTheLoopMiddleware in Agent Builder to require approval for tool calls.")
        return

    for approval in pending:
        tool_name = approval.get("tool_name", "unknown")
        args = approval.get("args", {})

        st.warning(f"**⚠️ Approval Required: {tool_name}**")
        st.json(args)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve", key=f"approve_{approval['id']}"):
                approval["status"] = "approved"
                st.session_state.approval_history.append(approval)
                st.session_state.pending_approvals.remove(approval)
                add_debug_event("APPROVAL", f"Approved: {tool_name}", "middleware")
                st.rerun()
        with col2:
            if st.button("✏️ Edit", key=f"edit_{approval['id']}"):
                st.session_state[f"editing_{approval['id']}"] = True
        with col3:
            if st.button("❌ Reject", key=f"reject_{approval['id']}"):
                approval["status"] = "rejected"
                st.session_state.approval_history.append(approval)
                st.session_state.pending_approvals.remove(approval)
                add_debug_event("APPROVAL", f"Rejected: {tool_name}", "middleware")
                st.rerun()


@st.fragment(run_every=2)  # Auto-refresh every 2 seconds
def concurrent_tasks_fragment():
    """Fragment for viewing concurrent/background tasks with live logs."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">🔄 Concurrent Tasks</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Monitor background task execution in real-time
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    exec_mgr = get_execution_manager()
    active_count = exec_mgr.get_active_count()
    active_tasks = exec_mgr.get_active_tasks_info()

    # Get any completed results
    completed_results = exec_mgr.get_completed_results()
    for result in completed_results:
        st.session_state.task_results.append(result)

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔄 Active", active_count)
    with col2:
        completed_count = len(st.session_state.task_results)
        st.metric("✅ Completed", completed_count)
    with col3:
        error_count = sum(1 for r in st.session_state.task_results if r.get("status") == "error")
        st.metric("❌ Errors", error_count)
    with col4:
        if st.button("🔄 Refresh Now", key="refresh_tasks"):
            st.rerun()

    st.divider()

    # ========== LIVE ACTIVE TASKS ==========
    if active_tasks:
        st.markdown("### 🔴 Live Execution")
        st.caption("Tasks currently running in background")

        for task in active_tasks:
            task_id = task.get("task_id", "unknown")
            prompt = task.get("prompt", "Unknown prompt")
            current_step = task.get("current_step", "initializing")
            elapsed_ms = task.get("elapsed_ms", 0)
            steps = task.get("steps", [])
            tool_calls = task.get("tool_calls", [])
            partial_response = task.get("partial_response", "")

            # Task card with live status
            with st.container():
                st.markdown(f"""
                <div style="border: 2px solid #2196F3; border-radius: 8px; padding: 15px; margin: 10px 0; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>🔵 Task {task_id}</strong>
                        <span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                            ⏱️ {elapsed_ms/1000:.1f}s
                        </span>
                    </div>
                    <div style="color: #666; font-size: 13px; margin-top: 5px;">
                        {prompt[:80]}{'...' if len(prompt) > 80 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Live progress
                col_step, col_tools = st.columns([2, 1])

                with col_step:
                    st.markdown(f"**Current Step:** `{current_step}`")

                    # Show step progress
                    if steps:
                        with st.expander(f"📋 Execution Steps ({len(steps)})", expanded=True):
                            for i, step in enumerate(steps[-5:]):  # Show last 5 steps
                                step_name = step.get("step", "unknown")
                                step_time = step.get("timestamp", "")[:19]
                                content_preview = step.get("content", "")[:100]

                                if step.get("tool_call"):
                                    tc = step["tool_call"]
                                    st.markdown(f"  `{i+1}.` 🔧 **{tc.get('name')}** - {step_time[11:]}")
                                elif content_preview:
                                    st.markdown(f"  `{i+1}.` 💬 Response chunk - {step_time[11:]}")
                                else:
                                    st.markdown(f"  `{i+1}.` ⚙️ {step_name} - {step_time[11:]}")

                with col_tools:
                    if tool_calls:
                        st.markdown(f"**Tool Calls:** {len(tool_calls)}")
                        for tc in tool_calls[-3:]:  # Show last 3 tool calls
                            st.code(f"🔧 {tc.get('name', 'unknown')}", language=None)

                # Partial response preview
                if partial_response:
                    with st.expander("💬 Partial Response Preview"):
                        st.markdown(partial_response[:300] + ("..." if len(partial_response) > 300 else ""))

                st.markdown("---")
    else:
        if active_count == 0:
            st.info("💤 No tasks currently running. Send a prompt in 'Background' mode to start a task.")

    # ========== COMPLETED TASKS ==========
    st.markdown("### 📋 Completed Tasks")

    if st.session_state.task_results:
        # Filter controls
        col_filter, col_sort = st.columns(2)
        with col_filter:
            filter_status = st.selectbox("Filter", ["All", "Completed", "Errors"], key="task_filter")
        with col_sort:
            sort_order = st.selectbox("Sort", ["Newest First", "Oldest First"], key="task_sort")

        # Filter and sort results
        results_to_show = st.session_state.task_results.copy()

        if filter_status == "Completed":
            results_to_show = [r for r in results_to_show if r.get("status") == "completed"]
        elif filter_status == "Errors":
            results_to_show = [r for r in results_to_show if r.get("status") == "error"]

        if sort_order == "Newest First":
            results_to_show = list(reversed(results_to_show))

        # Show results
        for result in results_to_show[-15:]:  # Limit to 15 most recent
            status = result.get("status", "unknown")
            task_id = result.get("task_id", "unknown")
            prompt = result.get("prompt", "Unknown")
            duration_ms = result.get("duration_ms", 0)
            started_at = result.get("started_at", "")
            completed_at = result.get("completed_at", "")

            # Status icon and color
            if status == "completed":
                icon = "✅"
                border_color = "#4CAF50"
            elif status == "error":
                icon = "❌"
                border_color = "#f44336"
            else:
                icon = "🔄"
                border_color = "#FF9800"

            header = f"{icon} **{task_id}** · {prompt[:40]}{'...' if len(prompt) > 40 else ''} · {duration_ms/1000:.2f}s"

            with st.expander(header):
                # Metadata
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.caption(f"**Started:** {started_at[11:19] if started_at else 'N/A'}")
                with meta_col2:
                    st.caption(f"**Completed:** {completed_at[11:19] if completed_at else 'N/A'}")
                with meta_col3:
                    st.caption(f"**Duration:** {duration_ms:.0f}ms")

                # Full prompt
                st.markdown("**Prompt:**")
                st.markdown(f"> {prompt}")

                # Response
                if result.get("message"):
                    st.markdown("**Response:**")
                    response = result["message"]
                    if len(response) > 1000:
                        st.markdown(response[:1000] + "...")
                        with st.popover("Show full response"):
                            st.markdown(response)
                    else:
                        st.markdown(response)

                # Error
                if result.get("error"):
                    st.error(f"**Error:** {result['error']}")
                    if result.get("traceback"):
                        with st.expander("🔍 Traceback"):
                            st.code(result["traceback"], language="python")

                # Tool calls
                if result.get("tool_calls"):
                    st.markdown(f"**🔧 Tool Calls ({len(result['tool_calls'])}):**")
                    for tc in result["tool_calls"]:
                        st.code(f"{tc.get('name', 'unknown')}: {json.dumps(tc.get('args', {}), indent=2)[:200]}")

                # Execution steps
                if result.get("steps"):
                    with st.expander(f"📋 Execution Steps ({len(result['steps'])})"):
                        for i, step in enumerate(result["steps"]):
                            step_name = step.get("step", "unknown")
                            step_time = step.get("timestamp", "")
                            st.caption(f"`{i+1}.` {step_name} - {step_time[11:19] if step_time else ''}")

        # Clear button
        def clear_task_results():
            st.session_state.task_results = []

        st.button("🗑️ Clear Results", key="clear_task_results", on_click=clear_task_results)
    else:
        st.info("📭 No completed tasks yet. Results will appear here after background tasks finish.")


@st.fragment
def scheduled_prompts_fragment():
    """Fragment for managing scheduled prompts."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">⏰ Scheduled Prompts</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Automate prompts to run at regular intervals
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    # Add new scheduled prompt
    with st.expander("➕ Add Scheduled Prompt", expanded=False):
        new_prompt = st.text_area("Prompt", key="new_scheduled_prompt", height=100,
                                   placeholder="Enter a prompt to run on schedule...")

        interval_options = {i.label: i for i in ScheduleInterval}
        selected_interval = st.selectbox(
            "Run Every",
            options=list(interval_options.keys()),
            index=2,  # Default to 1 minute
            key="scheduled_interval"
        )

        if st.button("Add Schedule", key="add_schedule"):
            if new_prompt.strip():
                schedule_id = str(uuid.uuid4())[:8]
                interval = interval_options[selected_interval]

                scheduled = ScheduledPrompt(
                    id=schedule_id,
                    prompt=new_prompt.strip(),
                    interval=interval,
                    enabled=True,
                    next_run=datetime.now() + timedelta(seconds=interval.seconds)
                )

                st.session_state.scheduled_prompts[schedule_id] = scheduled
                add_debug_event("SCHEDULE", f"Added: {new_prompt[:30]}... every {interval.label}", "scheduled")
                st.success(f"Added scheduled prompt (ID: {schedule_id})")
                st.rerun()
            else:
                st.warning("Please enter a prompt")

    # List scheduled prompts
    if st.session_state.scheduled_prompts:
        st.markdown("**Active Schedules:**")

        for schedule_id, scheduled in st.session_state.scheduled_prompts.items():
            status_class = "enabled" if scheduled.enabled else "disabled"
            status_icon = "🟢" if scheduled.enabled else "⚪"

            with st.expander(f"{status_icon} {scheduled.prompt[:40]}... ({scheduled.interval.label})"):
                st.markdown(f"**ID:** {schedule_id}")
                st.markdown(f"**Interval:** {scheduled.interval.label}")
                st.markdown(f"**Run Count:** {scheduled.run_count}")
                st.markdown(f"**Last Run:** {scheduled.last_run.strftime('%H:%M:%S') if scheduled.last_run else 'Never'}")
                st.markdown(f"**Next Run:** {scheduled.next_run.strftime('%H:%M:%S') if scheduled.next_run else 'Now'}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if scheduled.enabled:
                        if st.button("⏸️ Pause", key=f"pause_{schedule_id}"):
                            scheduled.enabled = False
                            add_debug_event("SCHEDULE", f"Paused: {schedule_id}", "scheduled")
                            st.rerun()
                    else:
                        if st.button("▶️ Resume", key=f"resume_{schedule_id}"):
                            scheduled.enabled = True
                            scheduled.next_run = datetime.now()
                            add_debug_event("SCHEDULE", f"Resumed: {schedule_id}", "scheduled")
                            st.rerun()

                with col2:
                    if st.button("🔄 Run Now", key=f"run_now_{schedule_id}"):
                        scheduled.next_run = datetime.now()
                        add_debug_event("SCHEDULE", f"Manual trigger: {schedule_id}", "scheduled")
                        st.rerun()

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{schedule_id}"):
                        del st.session_state.scheduled_prompts[schedule_id]
                        add_debug_event("SCHEDULE", f"Deleted: {schedule_id}", "scheduled")
                        st.rerun()

                # Show recent results
                if scheduled.results:
                    st.markdown("**Recent Results:**")
                    for res in reversed(scheduled.results[-5:]):
                        success_icon = "✅" if res.get("success") else "❌"
                        st.markdown(f"{success_icon} {res.get('timestamp', 'Unknown')[:19]} ({res.get('duration_ms', 0):.0f}ms)")
                        st.text(res.get("result", "No result")[:200])
    else:
        st.info("No scheduled prompts yet. Add one above.")


@st.fragment
def diagnostics_fragment():
    """Fragment for Ollama diagnostics."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">🏥 Ollama Diagnostics</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Test connections, load models, and debug agent configuration
        </p>
    </div>
    """, unsafe_allow_html=True)

    config = st.session_state.ollama_config

    # Show agent/session context
    render_agent_session_header()

    # Connection info card
    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div><strong>🌐 Endpoint:</strong> <code>{config.base_url}</code></div>
            <div><strong>🧠 Model:</strong> <code>{config.model}</code></div>
            <div><strong>🌡️ Temperature:</strong> {config.temperature}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model loading section
    st.markdown("### Model Loading")
    st.info("💡 **Tip:** If inference times out, load the model first. Model loading can take 1-5 minutes depending on model size and hardware.")

    # Model loading uses a container for dynamic updates
    load_status_container = st.container()
    col_load1, col_load2 = st.columns(2)

    with col_load1:
        if st.button("🔥 Load Model into Memory", key="load_model_diag", use_container_width=True):
            with load_status_container:
                status_text = st.empty()
                progress_bar = st.progress(0)

                def progress_callback(msg):
                    status_text.info(f"⏳ {msg}")
                    if "already loaded" in msg.lower():
                        progress_bar.progress(100)
                    elif "loading" in msg.lower():
                        progress_bar.progress(30)
                    elif "loaded" in msg.lower():
                        progress_bar.progress(100)

                client = OllamaClient(config)
                success = client.warmup_model(progress_callback=progress_callback)

                if success:
                    st.session_state.model_loaded = True
                    st.session_state.agent_instance = None
                    status_text.success("✅ Model loaded and ready!")
                    progress_bar.progress(100)
                else:
                    status_text.error("❌ Failed to load model")

    with col_load2:
        if st.button("🔍 Check Model Status", key="check_model_status", use_container_width=True):
            with load_status_container:
                client = OllamaClient(config)
                running = client.get_running_models()
                if running:
                    st.success(f"✅ Running models: {', '.join([m.get('name', 'unknown') for m in running])}")
                    st.session_state.model_loaded = True
                else:
                    st.warning("⚠️ No models currently loaded in memory")
                    st.session_state.model_loaded = False

    st.divider()

    # Quick diagnostics
    st.markdown("### Quick Diagnostics")
    if st.button("🔄 Run Quick Diagnostics", key="run_quick_diag"):
        results = {"connection": None, "model": None, "overall": "unknown"}
        progress = st.progress(0, "Testing connection...")

        conn = test_ollama_connection(config.base_url, timeout=10.0)
        results["connection"] = {"success": conn.success, "latency_ms": conn.latency_ms, "error": conn.error}
        progress.progress(50, "Testing model availability...")

        if conn.success:
            model = test_model_availability(config.base_url, config.model, timeout=10.0)
            results["model"] = {"success": model.success, "latency_ms": model.latency_ms, "error": model.error}
            results["overall"] = "ready" if model.success else "model_not_found"
        else:
            results["overall"] = "connection_failed"

        progress.progress(100, "Done!")
        st.session_state.ollama_connected = conn.success

        col1, col2 = st.columns(2)
        with col1:
            if results["connection"]["success"]:
                st.success(f"✅ Connection OK ({results['connection']['latency_ms']:.0f}ms)")
            else:
                st.error(f"❌ Connection: {results['connection']['error']}")
        with col2:
            if results["model"] and results["model"]["success"]:
                st.success(f"✅ Model available ({results['model']['latency_ms']:.0f}ms)")
            elif results["model"]:
                st.error(f"❌ Model: {results['model']['error']}")

    st.divider()

    # Agent Debug Info
    st.markdown("### 🔍 Agent Configuration Debug")
    st.caption("Shows exact parameters used when creating the agent")

    agent_info = st.session_state.last_agent_call_info
    if agent_info:
        # Summary
        st.markdown(f"""
**Last Agent Created:** {agent_info.get('timestamp', 'N/A')}

| Parameter | Value |
|-----------|-------|
| **Model** | `{agent_info['model']['name']}` |
| **Base URL** | `{agent_info['model']['base_url']}` |
| **Temperature** | `{agent_info['model']['temperature']}` |
| **Context Size** | `{agent_info['model']['num_ctx']}` |
| **Agent Name** | `{agent_info['agent']['name']}` |
| **Middleware** | {len(agent_info['middleware'])} active |
| **Tools** | {len(agent_info['tools'])} active |
| **Checkpointer** | `{agent_info['checkpointer']}` |
""")

        # Detailed view
        with st.expander("📋 Full Agent Parameters (JSON)", expanded=False):
            st.json(agent_info)

        with st.expander("📝 System Prompt", expanded=False):
            st.code(agent_info.get('system_prompt', 'No system prompt'), language="markdown")

        with st.expander("🔧 Active Middleware", expanded=False):
            for mw in agent_info.get('middleware', []):
                st.markdown(f"• `{mw}`")

        with st.expander("🛠️ Active Tools", expanded=False):
            for tool in agent_info.get('tools', []):
                st.markdown(f"• `{tool}`")

        # Copy as code
        with st.expander("💻 Agent Creation Code (Equivalent)", expanded=False):
            code = f'''from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Model configuration
llm = ChatOllama(
    base_url="{agent_info['model']['base_url']}",
    model="{agent_info['model']['name']}",
    temperature={agent_info['model']['temperature']},
    num_ctx={agent_info['model']['num_ctx']},
    num_predict={agent_info['model']['num_predict']},
)

# Middleware: {agent_info['middleware']}
# Tools: {agent_info['tools']}

agent = create_agent(
    model=llm,
    tools=[{', '.join(agent_info['tools'])}],
    middleware=[...],  # {len(agent_info['middleware'])} middleware
    system_prompt="""...""",  # See System Prompt expander
)
'''
            st.code(code, language="python")
    else:
        st.info("No agent has been created yet. Send a message to create an agent and see its configuration here.")

    # Force recreate agent button
    def reset_agent():
        st.session_state.agent_instance = None
        add_debug_event("AGENT", "Agent reset - will recreate on next message")

    st.button("🔄 Reset Agent (Recreate on Next Message)", key="reset_agent_btn", on_click=reset_agent)


# ============================================================================
# Chat Functions
# ============================================================================

def get_streaming_response(user_message: str, response_placeholder, status_placeholder) -> Dict[str, Any]:
    """Get streaming response from agent with step tracking."""
    st.session_state.current_phase = "sending"
    st.session_state.current_steps = []
    add_debug_event("SENDING", f"Request: {user_message[:40]}...")
    status_placeholder.markdown("🔄 **Creating agent...**")

    start_time = time.time()

    try:
        # Get or create agent
        agent = get_or_create_agent()

        thread_id = st.session_state.thread_id
        config = {"configurable": {"thread_id": thread_id}}
        context = st.session_state.user_context

        st.session_state.current_phase = "streaming"
        st.session_state.is_streaming = True
        status_placeholder.markdown("🟢 **Streaming response...**")

        full_response = []
        tool_calls_received = []
        todos_received = []
        tokens = 0
        first_token_time = None

        # Stream from agent
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            context=context,
            stream_mode="updates",
        ):
            for step_name, data in chunk.items():
                add_debug_event("STEP", f"{step_name}", "streaming")

                # Extract messages
                if "messages" in data:
                    for msg in data["messages"]:
                        # Content
                        if hasattr(msg, 'content') and msg.content:
                            content = str(msg.content)

                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft = (first_token_time - start_time) * 1000
                                add_debug_event("STREAMING", f"First content (TTFT: {ttft:.0f}ms)")
                                status_placeholder.markdown(f"🟢 **Streaming** (TTFT: {ttft:.0f}ms)")

                            full_response.append(content)
                            tokens += len(content.split())
                            response_placeholder.markdown("".join(full_response) + "▌")

                        # Tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_call = {
                                    "name": tc.get("name", "unknown"),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id", str(uuid.uuid4()))
                                }
                                tool_calls_received.append(tool_call)
                                add_tool_call(tool_call["name"], tool_call["args"])
                                add_step("tool", str(tc.get("args", {}))[:200], {"tool_name": tool_call["name"], "args": tool_call["args"]})

                        # Todos from additional_kwargs
                        if hasattr(msg, 'additional_kwargs'):
                            kwargs = msg.additional_kwargs
                            if 'todos' in kwargs:
                                todos_received = kwargs['todos']
                                update_todos_from_response(todos_received)

                # Check for tool results
                if step_name == "tools" and "messages" in data:
                    for msg in data["messages"]:
                        if hasattr(msg, 'content'):
                            # This is a tool result
                            add_debug_event("TOOL_RESULT", f"Result received", "tool")

        st.session_state.is_streaming = False
        total_time = time.time() - start_time
        tps = tokens / total_time if total_time > 0 else 0

        st.session_state.current_phase = "completed"
        st.session_state.model_loaded = True

        final = "".join(full_response)
        add_debug_event("COMPLETED", f"{tokens} tokens in {total_time:.1f}s ({tps:.1f} tok/s)")
        add_step("response", final)
        status_placeholder.empty()
        response_placeholder.markdown(final)

        return {
            "message": final,
            "duration_ms": total_time * 1000,
            "tokens_out": tokens,
            "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else 0,
            "tokens_per_second": tps,
            "tool_calls": tool_calls_received,
            "todos": todos_received,
        }

    except Exception as e:
        st.session_state.is_streaming = False
        st.session_state.current_phase = "error"
        error_msg = str(e)
        add_debug_event("ERROR", error_msg, "error", error=traceback.format_exc())
        logger.error(f"Streaming error: {error_msg}", exc_info=True)

        # Check if this is a connection error and provide detailed diagnostics
        is_connection_error = any(x in error_msg for x in [
            "10061", "Connection refused", "ConnectError",
            "timeout", "TimeoutError", "No connection",
            "actively refused", "Cannot connect"
        ])

        if is_connection_error:
            base_url = st.session_state.ollama_config.base_url
            diagnostic_msg = diagnose_connection_error(e, base_url)
            status_placeholder.error("❌ Connection Failed")
            response_placeholder.error(diagnostic_msg)
            # Mark Ollama as disconnected
            st.session_state.ollama_connected = False
            return {
                "message": f"Connection Error: {e}",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": diagnostic_msg
            }
        else:
            status_placeholder.error(f"❌ {e}")
            response_placeholder.error(f"**Error:** {e}")
            return {
                "message": f"Error: {e}",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": error_msg
            }


def send_background_prompt(prompt: str, callback: Optional[Callable] = None) -> str:
    """Send a prompt for background/non-blocking execution."""
    task_id = str(uuid.uuid4())[:8]

    exec_mgr = get_execution_manager()
    agent = get_or_create_agent()

    thread_id = st.session_state.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    context = st.session_state.user_context

    add_debug_event("BACKGROUND", f"Submitting task {task_id}: {prompt[:30]}...", "info")

    exec_mgr.submit_prompt(
        task_id=task_id,
        prompt=prompt,
        agent=agent,
        config=config,
        context=context,
        callback=callback
    )

    return task_id


def process_scheduled_prompts():
    """Process any scheduled prompts that are due."""
    now = datetime.now()

    for schedule_id, scheduled in st.session_state.scheduled_prompts.items():
        if scheduled.should_run():
            add_debug_event("SCHEDULE", f"Running scheduled: {schedule_id}", "scheduled")

            try:
                # Run in background
                def on_complete(result):
                    scheduled.update_after_run(result)

                send_background_prompt(scheduled.prompt, callback=on_complete)

                # Update next run time immediately to prevent duplicate runs
                scheduled.next_run = now + timedelta(seconds=scheduled.interval.seconds)

            except Exception as e:
                add_debug_event("SCHEDULE", f"Error running {schedule_id}: {e}", "error")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render simplified sidebar with connection and active agent info."""
    with st.sidebar:
        # App branding - Light theme colors
        st.markdown("""
        <div style="text-align: center; padding: 0.75rem 0 1rem 0;">
            <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                        width: 56px; height: 56px; border-radius: 14px; margin: 0 auto 0.75rem auto;
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);">
                <span style="font-size: 1.75rem;">🦜</span>
            </div>
            <h2 style="margin: 0; color: #1e293b; font-size: 1.25rem; font-weight: 700;">LangChain Agent</h2>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: #64748b;">Explorer & Debugger</p>
        </div>
        """, unsafe_allow_html=True)

        # Connection status section - compact card
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.875rem; margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">
                🔌 Connection
            </div>
        """, unsafe_allow_html=True)
        connection_status_fragment()
        st.markdown("</div>", unsafe_allow_html=True)

        # Active Agent Section - prominent card
        active_id = st.session_state.active_agent_id
        if active_id and active_id in st.session_state.saved_agents:
            agent = st.session_state.saved_agents[active_id]
            test_icon = "✅" if agent.get("test_passed") else "⚠️"
            mw_count = len(agent.get('middleware') or [])
            tool_count = len(agent.get('tools') or [])

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                        border: 1px solid #6ee7b7; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #047857; margin-bottom: 0.5rem; font-weight: 600;">
                    🤖 Active Agent
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="background: white; width: 40px; height: 40px; border-radius: 10px;
                                display: flex; align-items: center; justify-content: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <span style="font-size: 1.25rem;">🤖</span>
                    </div>
                    <div style="flex: 1;">
                        <strong style="color: #1e293b; font-size: 0.95rem;">{agent['name']}</strong> {test_icon}
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">
                            {agent.get('model', 'Unknown')[:20]}
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 0.75rem; margin-top: 0.75rem; font-size: 0.75rem; color: #059669;">
                    <span>🔧 {mw_count} middleware</span>
                    <span>🛠️ {tool_count} tools</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Deactivate button
            def deactivate_agent():
                st.session_state.active_agent_id = None
                st.session_state.agent_instance = None
                st.session_state.messages = []
                add_debug_event("AGENT", "Agent deactivated")

            st.button("❌ Deactivate", key="deactivate_agent", on_click=deactivate_agent, use_container_width=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px;
                        padding: 1.25rem; text-align: center; margin-bottom: 1rem;">
                <div style="background: #e2e8f0; width: 48px; height: 48px; border-radius: 12px;
                            margin: 0 auto 0.75rem auto; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 1.5rem; opacity: 0.5;">🤖</span>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.85rem; font-weight: 500;">No agent active</p>
                <p style="color: #94a3b8; margin: 0.25rem 0 0 0; font-size: 0.75rem;">Create one in Agent Builder</p>
            </div>
            """, unsafe_allow_html=True)

        # Quick Agent Selector (if there are saved agents)
        saved_agents = st.session_state.saved_agents
        if saved_agents:
            with st.expander("📚 Quick Select Agent", expanded=False):
                agent_options = {f"{a['name']} {'✅' if a.get('test_passed') else ''}": aid
                               for aid, a in saved_agents.items()}

                selected = st.selectbox(
                    "Select Agent",
                    options=["-- Select --"] + list(agent_options.keys()),
                    key="quick_agent_select",
                    label_visibility="collapsed"
                )

                if selected != "-- Select --":
                    def quick_activate():
                        aid = agent_options[selected]
                        agent = st.session_state.saved_agents[aid]
                        st.session_state.active_agent_id = aid
                        st.session_state.agent_config.name = agent["name"]
                        st.session_state.agent_config.description = agent.get("description", "")
                        st.session_state.agent_config.system_prompt = agent.get("system_prompt")
                        st.session_state.ollama_config.model = agent.get("model", "llama3.2:latest")
                        st.session_state.ollama_config.temperature = agent.get("temperature", 0.7)
                        st.session_state.ollama_config.num_ctx = agent.get("num_ctx", 8192)
                        agent_middleware = agent.get("middleware") or []
                        st.session_state.selected_middleware = [
                            MiddlewareType(m) for m in agent_middleware
                            if m in [mt.value for mt in MiddlewareType]
                        ]
                        st.session_state.selected_tools = agent.get("tools") or []
                        st.session_state.agent_instance = None
                        st.session_state.messages = []
                        add_debug_event("AGENT", f"Quick activated: {agent['name']}")

                    st.button("⚡ Activate", key="quick_activate_btn", on_click=quick_activate, use_container_width=True, type="primary")

        # Settings expander
        with st.expander("⚙️ Settings", expanded=False):
            # Connection settings
            if st.session_state.ollama_connected:
                st.markdown("**Connection**")
                base_url = st.text_input("Ollama URL", value=st.session_state.ollama_config.base_url, key="sidebar_url", label_visibility="collapsed")

                def apply_url():
                    st.session_state.ollama_config.base_url = base_url
                    st.session_state.ollama_connected = None
                    add_debug_event("CONFIG", f"URL updated: {base_url}")

                st.button("Apply URL", key="apply_url", on_click=apply_url, use_container_width=True)
                st.markdown("")

            # Context configuration
            st.markdown("**Context**")
            ctx = st.session_state.user_context
            ctx.project = st.text_input("Project", value=ctx.project, key="ctx_project", label_visibility="collapsed", placeholder="Project name")
            col_ctx1, col_ctx2 = st.columns(2)
            with col_ctx1:
                ctx.environment = st.selectbox("Env", ["development", "staging", "production"],
                                                index=["development", "staging", "production"].index(ctx.environment),
                                                label_visibility="collapsed", key="ctx_env")
            with col_ctx2:
                ctx.user_role = st.selectbox("Role", ["user", "admin", "developer"],
                                              index=["user", "admin", "developer"].index(ctx.user_role),
                                              label_visibility="collapsed", key="ctx_role")

        # Controls section at bottom
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">
            🎛️ Controls
        </div>
        """, unsafe_allow_html=True)

        st.session_state.debug_enabled = st.toggle("🔍 Debug Mode", value=st.session_state.debug_enabled, key="debug_toggle")

        # Session controls
        def clear_session():
            st.session_state.messages = []
            st.session_state.debug_events = []
            st.session_state.current_steps = []
            st.session_state.tool_calls = []
            st.session_state.todos = []
            st.session_state.current_phase = "idle"
            st.session_state.task_results = []
            save_current_conversation()
            add_debug_event("SESSION", "Chat cleared")

        def new_session():
            save_current_conversation()
            st.session_state.messages = []
            st.session_state.debug_events = []
            st.session_state.current_steps = []
            st.session_state.tool_calls = []
            st.session_state.todos = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.current_phase = "idle"
            st.session_state.agent_instance = None
            st.session_state.task_results = []
            st.session_state.last_agent_call_info = None
            st.session_state.active_conversation_id = None
            add_debug_event("SESSION", "New session started")

        col1, col2 = st.columns(2)
        with col1:
            st.button("🗑️ Clear", use_container_width=True, on_click=clear_session, key="clear_btn")
        with col2:
            st.button("🔄 New", use_container_width=True, on_click=new_session, key="new_btn")

        # Version info at very bottom
        st.markdown("""
        <div style="text-align: center; padding-top: 1rem; margin-top: 1rem; border-top: 1px solid #e2e8f0;">
            <span style="font-size: 0.7rem; color: #94a3b8;">v1.0 • LangChain + Ollama</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Conversation Management
# ============================================================================

def create_new_conversation(name: str = None, agent_id: str = None) -> str:
    """Create a new conversation and return its ID."""
    conv_id = str(uuid.uuid4())[:8]
    agent_id = agent_id or st.session_state.active_agent_id

    # Generate default name if not provided
    if not name:
        conv_count = len(st.session_state.conversations) + 1
        name = f"Conversation {conv_count}"

    st.session_state.conversations[conv_id] = {
        "id": conv_id,
        "name": name,
        "messages": [],
        "agent_id": agent_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "thread_id": str(uuid.uuid4()),
    }

    add_debug_event("CONVERSATION", f"Created new conversation: {name}")
    return conv_id


def switch_conversation(conv_id: str):
    """Switch to a different conversation."""
    if conv_id not in st.session_state.conversations:
        return

    # Save current conversation if active
    save_current_conversation()

    # Load the selected conversation
    conv = st.session_state.conversations[conv_id]
    st.session_state.active_conversation_id = conv_id
    st.session_state.messages = conv.get("messages", []).copy()
    st.session_state.thread_id = conv.get("thread_id", str(uuid.uuid4()))

    # Reset agent instance to use correct context
    st.session_state.agent_instance = None
    st.session_state.current_steps = []
    st.session_state.tool_calls = []
    st.session_state.todos = []

    add_debug_event("CONVERSATION", f"Switched to: {conv.get('name')}")


def save_current_conversation():
    """Save current messages to the active conversation."""
    conv_id = st.session_state.active_conversation_id
    if conv_id and conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]["messages"] = st.session_state.messages.copy()
        st.session_state.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
        st.session_state.conversations[conv_id]["thread_id"] = st.session_state.thread_id


def delete_conversation(conv_id: str):
    """Delete a conversation."""
    if conv_id in st.session_state.conversations:
        conv_name = st.session_state.conversations[conv_id].get("name", "Unknown")
        del st.session_state.conversations[conv_id]

        # If deleting active conversation, clear it
        if st.session_state.active_conversation_id == conv_id:
            st.session_state.active_conversation_id = None
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())

        add_debug_event("CONVERSATION", f"Deleted conversation: {conv_name}")


def rename_conversation(conv_id: str, new_name: str):
    """Rename a conversation."""
    if conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]["name"] = new_name
        st.session_state.conversations[conv_id]["updated_at"] = datetime.now().isoformat()


# ============================================================================
# Main Chat Interface
# ============================================================================

def render_chat_interface():
    """Render main chat interface."""
    # Check if an agent is active
    active_id = st.session_state.active_agent_id
    if not active_id or active_id not in st.session_state.saved_agents:
        # No agent active - show prompt to create/select one
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <span style="font-size: 5rem; opacity: 0.3;">🤖</span>
            <h2 style="color: #64748b; margin: 1rem 0;">No Agent Selected</h2>
            <p style="color: #94a3b8; max-width: 400px; margin: 0 auto 2rem auto;">
                Before you can chat, you need to create and activate an agent.
                Go to the <strong>Agent Builder</strong> tab to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick start options
        st.markdown("### 🚀 Quick Start")
        st.markdown("Create a pre-configured agent to get started quickly:")

        quick_cols = st.columns(3)
        quick_agents = [
            ("Coding Assistant", "Helps with coding, file operations, and shell commands",
             [MiddlewareType.TODOLIST, MiddlewareType.SHELL_TOOL, MiddlewareType.FILE_SEARCH],
             ["read_file", "write_file", "list_directory", "glob_search", "grep_search"]),
            ("Research Agent", "Focused on research and information gathering",
             [MiddlewareType.TODOLIST, MiddlewareType.FILE_SEARCH],
             ["read_file", "list_directory", "glob_search", "grep_search"]),
            ("Minimal Agent", "Basic agent with minimal tools",
             [MiddlewareType.TODOLIST],
             ["read_file", "list_directory"]),
        ]

        for i, (name, desc, middleware, tools) in enumerate(quick_agents):
            with quick_cols[i]:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                            padding: 1.25rem; text-align: center; height: 100%;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">{name}</h4>
                    <p style="color: #64748b; font-size: 0.85rem; margin: 0 0 1rem 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                def create_quick_agent(n=name, d=desc, m=middleware, t=tools):
                    agent_id = str(uuid.uuid4())[:8]
                    agent_data = {
                        "id": agent_id,
                        "name": n,
                        "description": d,
                        "system_prompt": "",
                        "middleware": [mw.value for mw in m],
                        "tools": t,
                        "model": st.session_state.ollama_config.model,
                        "temperature": 0.7,
                        "num_ctx": 8192,
                        "created_at": datetime.now().isoformat(),
                        "tested": False,
                        "test_passed": False,
                    }
                    st.session_state.saved_agents[agent_id] = agent_data
                    # Activate it
                    st.session_state.active_agent_id = agent_id
                    st.session_state.agent_config.name = n
                    st.session_state.agent_config.description = d
                    st.session_state.selected_middleware = m
                    st.session_state.selected_tools = t
                    st.session_state.agent_instance = None
                    add_debug_event("AGENT", f"Quick created and activated: {n}")

                st.button(f"Create {name}", key=f"quick_create_{i}",
                         on_click=create_quick_agent, use_container_width=True, type="primary")

        return  # Don't render the rest of chat interface

    # Agent is active - get its info
    active_agent = st.session_state.saved_agents[active_id]
    agent_config = st.session_state.agent_config
    exec_mgr = get_execution_manager()
    active_tasks = exec_mgr.get_active_count()

    # ===== Two-column layout: Conversations | Chat =====
    conv_col, chat_col = st.columns([1, 3])

    # ===== Left Column: Conversations Panel =====
    with conv_col:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #e2e8f0;
                    border-radius: 14px; padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.25rem;">💬</span>
                <h4 style="margin: 0; color: #1e293b; font-size: 1rem; font-weight: 600;">Conversations</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # New conversation button
        if st.button("➕ New Conversation", key="new_conv_btn", use_container_width=True, type="primary"):
            conv_id = create_new_conversation()
            st.session_state.active_conversation_id = conv_id
            st.session_state.messages = []
            st.session_state.thread_id = st.session_state.conversations[conv_id]["thread_id"]
            st.toast("✨ New conversation created!", icon="💬")
            st.rerun()

        st.divider()

        # List conversations
        conversations = st.session_state.conversations
        if not conversations:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 1rem; color: #94a3b8;">
                <span style="font-size: 2rem; opacity: 0.5;">📝</span>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">No conversations yet</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem;">Click "New Conversation" to start</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Sort by updated_at descending
            sorted_convs = sorted(
                conversations.items(),
                key=lambda x: x[1].get("updated_at", ""),
                reverse=True
            )

            for conv_id, conv in sorted_convs:
                is_active = st.session_state.active_conversation_id == conv_id
                msg_count = len(conv.get("messages", []))
                agent_id = conv.get("agent_id")
                agent_name = "Unknown"
                if agent_id and agent_id in st.session_state.saved_agents:
                    agent_name = st.session_state.saved_agents[agent_id].get("name", "Unknown")

                # Format timestamp
                updated = conv.get("updated_at", "")
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated)
                        time_str = dt.strftime("%H:%M")
                        date_str = dt.strftime("%b %d")
                    except:
                        time_str = ""
                        date_str = ""
                else:
                    time_str = ""
                    date_str = ""

                # Conversation card styling
                bg_color = "#e0f2fe" if is_active else "#ffffff"
                border_color = "#0ea5e9" if is_active else "#e2e8f0"

                st.markdown(f"""
                <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px;
                            padding: 0.75rem; margin-bottom: 0.5rem; cursor: pointer;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <strong style="color: #1e293b; font-size: 0.9rem;">{conv.get("name", "Unnamed")}</strong>
                        <span style="color: #94a3b8; font-size: 0.7rem;">{time_str}</span>
                    </div>
                    <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">
                        🤖 {agent_name} · {msg_count} msgs · {date_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Action buttons for this conversation
                btn_cols = st.columns([1, 1])
                with btn_cols[0]:
                    if st.button("📂", key=f"open_{conv_id}", help="Open conversation",
                                disabled=is_active, use_container_width=True):
                        switch_conversation(conv_id)
                        st.toast(f"💬 Switched to '{conv.get('name', 'Conversation')}'", icon="📂")
                        st.rerun()
                with btn_cols[1]:
                    if st.button("🗑️", key=f"del_{conv_id}", help="Delete conversation",
                                use_container_width=True):
                        conv_name = conv.get("name", "Conversation")
                        delete_conversation(conv_id)
                        st.toast(f"🗑️ Deleted '{conv_name}'", icon="🗑️")
                        st.rerun()
                        st.rerun()

    # ===== Right Column: Chat Interface =====
    with chat_col:
        # Active conversation header
        active_conv_id = st.session_state.active_conversation_id
        if active_conv_id and active_conv_id in st.session_state.conversations:
            conv = st.session_state.conversations[active_conv_id]
            conv_name = conv.get("name", "Unnamed")
            # Editable conversation name
            col_name, col_agent = st.columns([2, 1])
            with col_name:
                new_name = st.text_input(
                    "Conversation name",
                    value=conv_name,
                    key="edit_conv_name",
                    label_visibility="collapsed",
                    placeholder="Conversation name..."
                )
                if new_name != conv_name:
                    rename_conversation(active_conv_id, new_name)

            with col_agent:
                # Agent selector for this conversation
                agent_options = list(st.session_state.saved_agents.keys())
                agent_names = [st.session_state.saved_agents[aid].get("name", aid) for aid in agent_options]
                current_agent = conv.get("agent_id", active_id)
                if current_agent in agent_options:
                    current_idx = agent_options.index(current_agent)
                else:
                    current_idx = 0

                if agent_options:
                    selected_idx = st.selectbox(
                        "Use agent",
                        range(len(agent_options)),
                        format_func=lambda i: f"🤖 {agent_names[i]}",
                        index=current_idx,
                        key="conv_agent_select",
                        label_visibility="collapsed"
                    )
                    if agent_options[selected_idx] != current_agent:
                        st.session_state.conversations[active_conv_id]["agent_id"] = agent_options[selected_idx]
                        # Update active agent to match
                        new_agent_id = agent_options[selected_idx]
                        st.session_state.active_agent_id = new_agent_id
                        new_agent = st.session_state.saved_agents[new_agent_id]
                        st.session_state.agent_config.name = new_agent.get("name", "")
                        st.session_state.agent_config.description = new_agent.get("description", "")
                        # Load middleware
                        mw_values = new_agent.get("middleware") or []
                        st.session_state.selected_middleware = [
                            MiddlewareType(mw) for mw in mw_values
                            if mw in [m.value for m in MiddlewareType]
                        ]
                        st.session_state.selected_tools = new_agent.get("tools") or []
                        st.session_state.agent_instance = None
                        st.rerun()
        else:
            # No active conversation - prompt to create one
            st.info("👆 Select a conversation from the left panel or create a new one to start chatting.")

        # Connection status indicator
        if st.session_state.ollama_connected:
            conn_class = "connected"
            conn_icon = "🟢"
            conn_text = "Connected"
        elif st.session_state.ollama_connected is None:
            conn_class = "unknown"
            conn_icon = "⚪"
            conn_text = "Not Checked"
        else:
            conn_class = "disconnected"
            conn_icon = "🔴"
            conn_text = "Disconnected"

        # Test status indicator
        test_status = ""
        if not active_agent.get("tested"):
            test_status = "<span style='background: #fef3c7; color: #92400e; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem;'>⚠️ Not Tested</span>"
        elif active_agent.get("test_passed"):
            test_status = "<span style='background: #dcfce7; color: #166534; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem;'>✅ Tested</span>"
        else:
            test_status = "<span style='background: #fee2e2; color: #991b1b; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem;'>❌ Test Failed</span>"

        # Enhanced status bar
        st.markdown(f"""
        <div class="status-bar">
            <div class="status-item"><strong>🤖 Agent:</strong> {agent_config.name} {test_status}</div>
            <div class="status-item"><strong>🔧 Middleware:</strong> {len(st.session_state.selected_middleware)}</div>
            <div class="status-item"><strong>🧵 Thread:</strong> {st.session_state.thread_id[:8]}...</div>
            <div class="connection-status {conn_class}">{conn_icon} {conn_text}</div>
            {"<div class='live-indicator'>🔴 " + str(active_tasks) + " Running</div>" if active_tasks > 0 else ""}
        </div>
        """, unsafe_allow_html=True)

        # Debug panel
        if st.session_state.debug_enabled:
            with st.expander("🔍 Debug Panel", expanded=False):
                debug_panel_fragment()

        st.divider()

        # Process scheduled prompts
        process_scheduled_prompts()

        # Check for completed background tasks
        exec_mgr = get_execution_manager()
        completed = exec_mgr.get_completed_results()
        for result in completed:
            st.session_state.task_results.append(result)
            if result.get("status") == "completed":
                add_debug_event("BACKGROUND", f"Task completed: {result.get('task_id')}", "info")

        # Chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # Display errors with error styling, normal messages with markdown
                if msg.get("error"):
                    error_content = msg.get("error", "Unknown error")
                    # Check if it's a detailed diagnostic (contains markdown)
                    if "**" in error_content or "```" in error_content:
                        st.markdown(error_content)
                    else:
                        st.error(f"❌ {error_content}")
                else:
                    st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("duration_ms") and not msg.get("error"):
                    parts = []
                    if msg.get("ttft_ms"):
                        parts.append(f"TTFT: {msg['ttft_ms']:.0f}ms")
                    if msg.get("tokens_per_second"):
                        parts.append(f"{msg['tokens_per_second']:.1f} tok/s")
                    if msg.get("tokens_out"):
                        parts.append(f"{msg['tokens_out']} tokens")
                    if msg.get("tool_calls"):
                        parts.append(f"{len(msg['tool_calls'])} tool calls")
                    if parts:
                        st.caption(" | ".join(parts))

        # Chat input with mode selection
        col_input, col_mode = st.columns([5, 1])

        with col_mode:
            send_mode = st.selectbox(
                "Mode",
                ["Blocking", "Background"],
                key="send_mode",
                label_visibility="collapsed"
            )

        with col_input:
            if prompt := st.chat_input("Message..."):
                # Create conversation if none exists
                if not st.session_state.active_conversation_id:
                    conv_id = create_new_conversation()
                    st.session_state.active_conversation_id = conv_id

                # Quick connection check before sending
                if st.session_state.ollama_connected is False:
                    base_url = st.session_state.ollama_config.base_url
                    st.error(f"❌ **Ollama not connected** to `{base_url}`")
                    st.info("💡 **Tips:**\n"
                            "1. Start Ollama: `ollama serve`\n"
                            "2. Click 🔄 in sidebar to reconnect\n"
                            "3. Check the Diagnostics tab for details")
                    return

                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat(),
                })

                with st.chat_message("user"):
                    st.markdown(prompt)

                if send_mode == "Background":
                    # Non-blocking execution
                    task_id = send_background_prompt(prompt)
                    st.info(f"✅ Prompt submitted (Task: {task_id}). Check 'Concurrent Tasks' tab for results.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"*Processing in background (Task: {task_id})...*",
                        "timestamp": datetime.now().isoformat(),
                        "background_task": task_id,
                    })
                else:
                    # Blocking execution
                    with st.chat_message("assistant"):
                        response_ph = st.empty()
                        status_ph = st.empty()
                        response = get_streaming_response(prompt, response_ph, status_ph)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("message", ""),
                        "duration_ms": response.get("duration_ms", 0),
                        "tokens_out": response.get("tokens_out", 0),
                        "ttft_ms": response.get("ttft_ms", 0),
                        "tokens_per_second": response.get("tokens_per_second", 0),
                        "tool_calls": response.get("tool_calls", []),
                        "timestamp": datetime.now().isoformat(),
                        "error": response.get("error"),
                    })

                    # Record to history
                    tracker = get_interaction_tracker()
                    iid = str(uuid.uuid4())
                    tracker.start_interaction(iid, agent_config.name, "langchain",
                                              prompt, "memory", [m.value for m in st.session_state.selected_middleware])

                    # Record tool calls
                    for tc in response.get("tool_calls", []):
                        tracker.record_tool_call(
                            tc.get("id", str(uuid.uuid4())),
                            tc.get("name", "unknown"),
                            tc.get("args", {}),
                            result=None,
                            success=True
                        )

                    # Record todos
                    for todo in response.get("todos", []):
                        tracker.record_todo(
                            todo.get("content", "Unknown"),
                            todo.get("status", "pending"),
                            agent_config.name
                        )

                    tracker.complete_interaction(
                        response.get("message", ""),
                        response.get("duration_ms", 0),
                        0,
                        response.get("tokens_out", 0),
                        response.get("error")
                    )

                # Save messages to active conversation
                save_current_conversation()

                st.rerun()


# ============================================================================
# Agent Builder
# ============================================================================

@st.fragment
def agent_builder_fragment():
    """Fragment for creating and configuring agents."""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #1e293b;">🔧 Agent Builder</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Create, configure, and test your agents before using them in chat
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get builder config
    builder = st.session_state.builder_agent_config

    # Layout: Left side for config, right side for saved agents
    col_build, col_saved = st.columns([2, 1])

    with col_build:
        st.markdown("### 📝 Agent Configuration")

        # Basic Info
        st.markdown("#### Basic Information")
        builder["name"] = st.text_input(
            "Agent Name",
            value=builder.get("name", "New Agent"),
            key="builder_name",
            placeholder="Give your agent a name..."
        )
        builder["description"] = st.text_area(
            "Description",
            value=builder.get("description", "A helpful AI assistant"),
            key="builder_desc",
            height=80,
            placeholder="Describe what this agent does..."
        )

        # Model Configuration
        st.markdown("#### 🧠 Model Configuration")
        model_col1, model_col2 = st.columns(2)

        with model_col1:
            # Model selection
            models = st.session_state.available_models
            model_names = [m.get("name", "") for m in models] if models else []
            curr_model = builder.get("model", "llama3.2:latest")

            if curr_model and curr_model not in model_names:
                model_names.insert(0, curr_model)

            if model_names:
                try:
                    curr_idx = model_names.index(curr_model)
                except ValueError:
                    curr_idx = 0
                builder["model"] = st.selectbox(
                    "Model",
                    model_names,
                    index=curr_idx,
                    key="builder_model"
                )
            else:
                builder["model"] = st.text_input(
                    "Model",
                    value=curr_model,
                    key="builder_model_input",
                    placeholder="e.g., llama3.2:latest"
                )

        with model_col2:
            builder["temperature"] = st.slider(
                "Temperature",
                0.0, 2.0,
                builder.get("temperature", 0.7),
                0.1,
                key="builder_temp"
            )

        builder["num_ctx"] = st.select_slider(
            "Context Window",
            options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            value=builder.get("num_ctx", 8192),
            key="builder_ctx"
        )

        # System Prompt
        st.markdown("#### 📜 System Prompt")
        builder["system_prompt"] = st.text_area(
            "System Prompt (optional)",
            value=builder.get("system_prompt", ""),
            key="builder_system_prompt",
            height=120,
            placeholder="Custom instructions for the agent... Leave empty for default.",
            label_visibility="collapsed"
        )

        # Middleware Selection
        st.markdown("#### 🔧 Middleware")
        st.caption("Select middleware components to enhance agent capabilities")

        configs = get_middleware_configs()
        # Defensive: ensure middleware is always a list, never None
        current_middleware = builder.get("middleware") or []
        if not isinstance(current_middleware, list):
            current_middleware = []

        # Group middleware by category
        mw_cols = st.columns(2)
        for i, cfg in enumerate(sorted(configs, key=lambda x: x.priority)):
            with mw_cols[i % 2]:
                enabled = cfg.middleware_type in current_middleware
                if st.checkbox(
                    f"{cfg.name}",
                    value=enabled,
                    key=f"builder_mw_{cfg.name}",
                    help=cfg.description
                ):
                    if cfg.middleware_type not in current_middleware:
                        current_middleware.append(cfg.middleware_type)
                else:
                    if cfg.middleware_type in current_middleware:
                        current_middleware.remove(cfg.middleware_type)

        builder["middleware"] = current_middleware

        # Tools Selection
        st.markdown("#### 🛠️ Tools")
        tool_registry = get_tool_registry()
        tool_info = tool_registry.get_tool_info()
        available_tools = list(tool_info.keys())
        # Defensive: ensure tools is always a list
        current_tools = builder.get("tools") or []
        if not isinstance(current_tools, list):
            current_tools = []

        builder["tools"] = st.multiselect(
            "Select Tools",
            options=available_tools,
            default=[t for t in current_tools if t in available_tools],
            key="builder_tools"
        )

        st.markdown("---")

        # Action Buttons
        st.markdown("### 💾 Save Agent")
        action_cols = st.columns([2, 1, 1])

        with action_cols[0]:
            save_name = st.text_input(
                "Save as",
                value=builder.get("name", "New Agent"),
                key="save_agent_name",
                label_visibility="collapsed",
                placeholder="Agent name to save..."
            )

        with action_cols[1]:
            def save_agent():
                agent_id = str(uuid.uuid4())[:8]
                agent_data = {
                    "id": agent_id,
                    "name": save_name,
                    "description": builder.get("description", ""),
                    "system_prompt": builder.get("system_prompt", ""),
                    "middleware": [m.value for m in builder.get("middleware", [])],
                    "tools": builder.get("tools", []),
                    "model": builder.get("model", "llama3.2:latest"),
                    "temperature": builder.get("temperature", 0.7),
                    "num_ctx": builder.get("num_ctx", 8192),
                    "created_at": datetime.now().isoformat(),
                    "tested": False,
                    "test_passed": False,
                }
                st.session_state.saved_agents[agent_id] = agent_data
                add_debug_event("AGENT", f"Saved agent: {save_name} ({agent_id})")
                st.toast(f"✅ Agent '{save_name}' saved!", icon="💾")

            st.button("💾 Save", key="save_agent_btn", on_click=save_agent, use_container_width=True, type="primary")

        with action_cols[2]:
            def test_agent():
                """Test the current configuration with a simple prompt."""
                st.session_state.current_phase = "testing"
                add_debug_event("AGENT", "Testing agent configuration...")

            st.button("🧪 Test", key="test_agent_btn", on_click=test_agent, use_container_width=True)

    # Right column: Saved Agents
    with col_saved:
        st.markdown("### 📚 Saved Agents")

        saved = st.session_state.saved_agents
        if not saved:
            st.info("No saved agents yet. Configure and save an agent to get started.")
        else:
            for agent_id, agent_data in saved.items():
                is_active = st.session_state.active_agent_id == agent_id
                test_status = "✅" if agent_data.get("test_passed") else ("🧪" if agent_data.get("tested") else "⚪")

                # Agent card
                card_style = "border: 2px solid #22c55e;" if is_active else "border: 1px solid #e2e8f0;"
                st.markdown(f"""
                <div style="{card_style} border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem; background: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{agent_data['name']}</strong>
                        <span>{test_status}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">
                        {agent_data.get('model', 'Unknown')} · {len(agent_data.get('middleware', []))} middleware · {len(agent_data.get('tools', []))} tools
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_actions = st.columns(3)
                with col_actions[0]:
                    def activate_agent(aid=agent_id):
                        st.session_state.active_agent_id = aid
                        agent = st.session_state.saved_agents[aid]
                        st.session_state.agent_config.name = agent["name"]
                        st.session_state.agent_config.description = agent.get("description", "")
                        st.session_state.agent_config.system_prompt = agent.get("system_prompt")
                        st.session_state.ollama_config.model = agent.get("model", "llama3.2:latest")
                        st.session_state.ollama_config.temperature = agent.get("temperature", 0.7)
                        st.session_state.ollama_config.num_ctx = agent.get("num_ctx", 8192)
                        agent_middleware = agent.get("middleware") or []
                        st.session_state.selected_middleware = [
                            MiddlewareType(m) for m in agent_middleware
                            if m in [mt.value for mt in MiddlewareType]
                        ]
                        st.session_state.selected_tools = agent.get("tools") or []
                        st.session_state.agent_instance = None
                        st.session_state.messages = []
                        add_debug_event("AGENT", f"Activated: {agent['name']}")
                        st.toast(f"🤖 Agent '{agent['name']}' activated!", icon="✅")

                    btn_label = "✓ Active" if is_active else "⚡ Activate"
                    st.button(btn_label, key=f"activate_{agent_id}", on_click=activate_agent,
                             use_container_width=True, disabled=is_active, type="primary" if not is_active else "secondary")

                with col_actions[1]:
                    def load_to_builder(aid=agent_id):
                        agent = st.session_state.saved_agents[aid]
                        # Defensive: handle None middleware/tools
                        agent_middleware = agent.get("middleware") or []
                        agent_tools = agent.get("tools") or []
                        st.session_state.builder_agent_config = {
                            "name": agent["name"],
                            "description": agent.get("description", ""),
                            "system_prompt": agent.get("system_prompt", ""),
                            "middleware": [
                                MiddlewareType(m) for m in agent_middleware
                                if m in [mt.value for mt in MiddlewareType]
                            ],
                            "tools": agent_tools,
                            "model": agent.get("model", "llama3.2:latest"),
                            "temperature": agent.get("temperature", 0.7),
                            "num_ctx": agent.get("num_ctx", 8192),
                        }

                    st.button("📝 Edit", key=f"edit_{agent_id}", on_click=load_to_builder, use_container_width=True)

                with col_actions[2]:
                    def delete_agent(aid=agent_id):
                        agent_name = st.session_state.saved_agents[aid].get("name", "Agent")
                        del st.session_state.saved_agents[aid]
                        if st.session_state.active_agent_id == aid:
                            st.session_state.active_agent_id = None
                            st.session_state.agent_instance = None
                        add_debug_event("AGENT", f"Deleted agent: {aid}")
                        st.toast(f"🗑️ Agent '{agent_name}' deleted", icon="🗑️")

                    st.button("🗑️", key=f"delete_{agent_id}", on_click=delete_agent, use_container_width=True)

                st.markdown("")  # Spacing

        # Quick create from preset
        st.markdown("---")
        st.markdown("#### 🚀 Quick Start")
        preset_options = {
            "Coding Assistant": {
                "name": "Coding Assistant",
                "description": "Helps with coding tasks, file operations, and shell commands",
                "middleware": [MiddlewareType.TODOLIST, MiddlewareType.SHELL_TOOL, MiddlewareType.FILE_SEARCH],
                "tools": ["read_file", "write_file", "list_directory", "glob_search", "grep_search"],
            },
            "Research Agent": {
                "name": "Research Agent",
                "description": "Focused on research and information gathering",
                "middleware": [MiddlewareType.TODOLIST, MiddlewareType.FILE_SEARCH],
                "tools": ["read_file", "list_directory", "glob_search", "grep_search"],
            },
            "Minimal Agent": {
                "name": "Minimal Agent",
                "description": "Basic agent with minimal tools",
                "middleware": [MiddlewareType.TODOLIST],
                "tools": ["read_file", "list_directory"],
            },
        }

        for preset_name, preset_data in preset_options.items():
            def apply_preset(data=preset_data):
                st.session_state.builder_agent_config.update(data)

            if st.button(f"📋 {preset_name}", key=f"preset_{preset_name}", use_container_width=True):
                apply_preset()
                st.rerun()


# ============================================================================
# Agent Test Fragment
# ============================================================================

@st.fragment
def agent_test_fragment():
    """Test an agent configuration with a simple prompt."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">🧪 Agent Testing</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Verify your agent works correctly before using it in chat
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if there's an agent to test
    active_id = st.session_state.active_agent_id
    if not active_id:
        st.warning("No agent selected. Go to Agent Builder and activate an agent first.")
        return

    agent_data = st.session_state.saved_agents.get(active_id)
    if not agent_data:
        st.error("Selected agent not found.")
        return

    # Show agent info with better styling
    mw_count = len(agent_data.get('middleware') or [])
    tool_count = len(agent_data.get('tools') or [])
    test_status_icon = "✅" if agent_data.get("test_passed") else ("🧪" if agent_data.get("tested") else "⬜")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #93c5fd;
                border-radius: 12px; padding: 1.25rem; margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <div style="background: white; width: 44px; height: 44px; border-radius: 10px;
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <span style="font-size: 1.5rem;">🤖</span>
            </div>
            <div>
                <strong style="color: #1e293b; font-size: 1.1rem;">{agent_data['name']}</strong> {test_status_icon}
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">
                    {agent_data.get('model', 'Unknown')}
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 1rem; font-size: 0.8rem; color: #3b82f6; margin-top: 0.5rem;">
            <span>🔧 {mw_count} middleware</span>
            <span>🛠️ {tool_count} tools</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Test prompts
    st.markdown("#### Test Prompts")
    test_prompts = [
        "Hello! Can you introduce yourself and tell me what you can do?",
        "What tools do you have access to?",
        "Create a simple todo list with 3 items for testing purposes.",
    ]

    selected_test = st.selectbox(
        "Select a test prompt or enter custom",
        ["Custom..."] + test_prompts,
        key="test_prompt_select"
    )

    if selected_test == "Custom...":
        test_prompt = st.text_area("Custom Test Prompt", key="custom_test_prompt", height=80)
    else:
        test_prompt = selected_test

    # Run test
    test_container = st.container()

    def run_test():
        if not test_prompt:
            return

        with test_container:
            st.markdown("---")
            st.markdown("#### Test Results")

            response_ph = st.empty()
            status_ph = st.empty()

            status_ph.info("🔄 Running test...")

            try:
                response = get_streaming_response(test_prompt, response_ph, status_ph)

                if response.get("error"):
                    st.error(f"Test failed: {response['error']}")
                    agent_data["tested"] = True
                    agent_data["test_passed"] = False
                    st.toast("❌ Test failed", icon="⚠️")
                else:
                    duration = response.get('duration_ms', 0)/1000
                    st.success(f"✅ Test passed! ({duration:.2f}s)")
                    agent_data["tested"] = True
                    agent_data["test_passed"] = True
                    st.toast(f"✅ Test passed in {duration:.1f}s!", icon="🎉")

            except Exception as e:
                st.error(f"Test error: {e}")
                agent_data["tested"] = True
                agent_data["test_passed"] = False
                st.toast("❌ Test error occurred", icon="⚠️")

    st.button("▶️ Run Test", key="run_test_btn", on_click=run_test, type="primary", use_container_width=True)

    # Show test history
    if agent_data.get("tested"):
        st.markdown("---")
        if agent_data.get("test_passed"):
            st.success("✅ This agent has passed testing and is ready to use in Chat!")
        else:
            st.warning("⚠️ This agent has failed testing. Review the configuration and try again.")


# ============================================================================
# Middleware Explorer
# ============================================================================

@st.fragment
def middleware_explorer_fragment():
    """Fragment for middleware exploration and configuration."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e293b;">⚙️ Middleware Explorer</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            14 built-in LangChain middleware types for enhanced agent capabilities
        </p>
    </div>
    """, unsafe_allow_html=True)

    configs = get_middleware_configs()
    for cfg in sorted(configs, key=lambda x: x.priority):
        enabled = cfg.middleware_type in st.session_state.selected_middleware
        icon = "✅" if enabled else "⬜"

        with st.expander(f"{icon} **{cfg.name}** (Priority: {cfg.priority})"):
            st.markdown(cfg.description)

            # Show default options
            st.markdown("**Default Options:**")
            st.json(cfg.options)

            col1, col2 = st.columns(2)
            with col1:
                if enabled:
                    if st.button(f"Disable", key=f"disable_{cfg.name}"):
                        st.session_state.selected_middleware.remove(cfg.middleware_type)
                        st.session_state.agent_instance = None
                        add_debug_event("CONFIG", f"Disabled: {cfg.name}", "middleware")
                        st.rerun()
                else:
                    if st.button(f"Enable", key=f"enable_{cfg.name}"):
                        st.session_state.selected_middleware.append(cfg.middleware_type)
                        st.session_state.agent_instance = None
                        add_debug_event("CONFIG", f"Enabled: {cfg.name}", "middleware")
                        st.rerun()


@st.fragment
def history_fragment():
    """Fragment for history dashboard with charts and visualizations."""
    store = get_history_store()
    stats = store.get_statistics()

    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #1e293b;">📊 Analytics Dashboard</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
            Track performance, usage patterns, and agent interactions
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show agent/session context
    render_agent_session_header()

    # ========== KEY METRICS ROW ==========
    col1, col2, col3, col4, col5 = st.columns(5)

    total_interactions = stats.get("total_interactions", 0)
    success_rate = stats.get("success_rate", 100)
    avg_duration = stats.get("avg_duration_ms", 0) / 1000
    total_tokens = stats.get("total_tokens_in", 0) + stats.get("total_tokens_out", 0)
    last_24h = stats.get("interactions_last_24h", 0)

    with col1:
        st.metric("📝 Total Interactions", f"{total_interactions:,}",
                 delta=f"+{last_24h} today" if last_24h > 0 else None)
    with col2:
        color = "normal" if success_rate >= 90 else "inverse"
        st.metric("✅ Success Rate", f"{success_rate:.1f}%",
                 delta=f"{stats.get('total_errors', 0)} errors" if stats.get('total_errors', 0) > 0 else None,
                 delta_color=color)
    with col3:
        st.metric("⏱️ Avg Response", f"{avg_duration:.2f}s")
    with col4:
        st.metric("🎟️ Total Tokens", f"{total_tokens:,}")
    with col5:
        st.metric("🔧 Tool Calls", f"{stats.get('total_tool_calls', 0):,}")

    st.divider()

    # ========== CHARTS SECTION ==========
    chart_tabs = st.tabs(["📈 Activity", "🔧 Tools", "⏱️ Performance", "🤖 Agents"])

    # ----- Activity Timeline -----
    with chart_tabs[0]:
        st.markdown("#### Interaction Activity (Last 7 Days)")

        timeline = store.get_interactions_timeline(days=7)
        if timeline:
            # Prepare data for charts

            df_timeline = pd.DataFrame(timeline)
            df_timeline['date'] = pd.to_datetime(df_timeline['date'])

            # Activity bar chart
            chart_data = df_timeline.set_index('date')[['interactions', 'errors']]
            chart_data.columns = ['Successful', 'Errors']
            st.bar_chart(chart_data, color=["#4CAF50", "#f44336"])

            # Stats below chart
            col1, col2, col3 = st.columns(3)
            with col1:
                total_week = df_timeline['interactions'].sum()
                st.metric("This Week", f"{total_week} interactions")
            with col2:
                avg_daily = df_timeline['interactions'].mean()
                st.metric("Daily Average", f"{avg_daily:.1f}")
            with col3:
                peak_day = df_timeline.loc[df_timeline['interactions'].idxmax()]
                st.metric("Peak Day", f"{peak_day['date'].strftime('%a %m/%d')}: {int(peak_day['interactions'])}")
        else:
            st.info("📭 No activity data yet. Start chatting to see analytics!")

        # Hourly activity
        st.markdown("#### Today's Hourly Activity")
        hourly = store.get_hourly_activity(days=1)
        if hourly:
            df_hourly = pd.DataFrame(hourly)
            st.bar_chart(df_hourly.set_index('hour')['interactions'], color="#2196F3")
        else:
            st.caption("No activity recorded today")

    # ----- Tool Usage -----
    with chart_tabs[1]:
        st.markdown("#### Tool Usage Distribution")

        tool_usage = stats.get("tool_usage", {})
        if tool_usage:

            # Tool calls pie chart data
            tool_names = list(tool_usage.keys())
            tool_counts = [tool_usage[t]["count"] for t in tool_names]

            df_tools = pd.DataFrame({
                "Tool": tool_names,
                "Calls": tool_counts,
                "Avg Duration (ms)": [tool_usage[t]["avg_duration_ms"] or 0 for t in tool_names],
                "Success Rate": [tool_usage[t]["success_rate"] for t in tool_names],
            })

            # Bar chart of tool usage
            st.bar_chart(df_tools.set_index('Tool')['Calls'], color="#9C27B0")

            # Detailed table
            st.markdown("#### Tool Details")
            for _, row in df_tools.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**🔧 {row['Tool']}**")
                with col2:
                    st.metric("Calls", int(row['Calls']), label_visibility="collapsed")
                with col3:
                    st.metric("Avg", f"{row['Avg Duration (ms)']:.0f}ms", label_visibility="collapsed")
                with col4:
                    rate = row['Success Rate']
                    color = "🟢" if rate >= 90 else "🟡" if rate >= 70 else "🔴"
                    st.markdown(f"{color} {rate:.0f}%")
        else:
            st.info("🔧 No tool usage recorded yet")

    # ----- Performance -----
    with chart_tabs[2]:
        st.markdown("#### Response Time Distribution")

        distribution = store.get_response_time_distribution()
        if distribution:

            df_dist = pd.DataFrame(distribution)
            st.bar_chart(df_dist.set_index('bucket')['count'], color="#FF9800")

            # Performance stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⚡ Fastest", f"{stats.get('min_duration_ms', 0)/1000:.2f}s")
            with col2:
                st.metric("📊 Average", f"{stats.get('avg_duration_ms', 0)/1000:.2f}s")
            with col3:
                st.metric("🐢 Slowest", f"{stats.get('max_duration_ms', 0)/1000:.2f}s")

            # Token efficiency
            st.markdown("#### Token Usage")
            if total_interactions > 0:
                avg_tokens_per_interaction = total_tokens / total_interactions
                st.metric("Avg Tokens/Interaction", f"{avg_tokens_per_interaction:.0f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Input Tokens", f"{stats.get('total_tokens_in', 0):,}")
                with col2:
                    st.metric("Output Tokens", f"{stats.get('total_tokens_out', 0):,}")
        else:
            st.info("⏱️ No performance data yet")

    # ----- Agents -----
    with chart_tabs[3]:
        st.markdown("#### Agent Usage")

        by_agent = stats.get("by_agent_name", {})
        if by_agent:

            df_agents = pd.DataFrame([
                {
                    "Agent": name,
                    "Interactions": data["count"],
                    "Avg Duration (s)": (data["avg_duration_ms"] or 0) / 1000,
                    "Total Tokens": data.get("total_tokens", 0),
                }
                for name, data in by_agent.items()
            ])

            st.bar_chart(df_agents.set_index('Agent')['Interactions'], color="#00BCD4")

            st.markdown("#### Agent Details")
            st.dataframe(df_agents, use_container_width=True, hide_index=True)
        else:
            st.info("🤖 No agent usage data yet")

    st.divider()

    # ========== INTERACTION HISTORY ==========
    st.markdown("### 📜 Interaction History")

    # Search and filter
    col_search, col_filter, col_limit = st.columns([3, 2, 1])

    with col_search:
        search_query = st.text_input("🔍 Search messages", placeholder="Search in user or assistant messages...",
                                     key="history_search")

    with col_filter:
        filter_options = ["All", "With Tools", "With Errors", "No Errors"]
        selected_filter = st.selectbox("Filter", filter_options, key="history_filter")

    with col_limit:
        limit = st.selectbox("Show", [10, 25, 50, 100], index=1, key="history_limit")

    # Get interactions
    if search_query:
        interactions = store.search_interactions(search_query, limit=limit)
        st.caption(f"🔍 Found {len(interactions)} results for '{search_query}'")
    else:
        interactions = store.get_interactions(limit=limit)

    # Apply filters
    if selected_filter == "With Tools":
        interactions = [i for i in interactions if i.get("tool_calls")]
    elif selected_filter == "With Errors":
        interactions = [i for i in interactions if i.get("error")]
    elif selected_filter == "No Errors":
        interactions = [i for i in interactions if not i.get("error")]

    if not interactions:
        st.info("📭 No interactions found. Start chatting to build history!")
    else:
        for i in interactions:
            # Determine status icon
            if i.get("error"):
                status_icon = "❌"
                status_color = "#f44336"
            elif i.get("tool_calls"):
                status_icon = "🔧"
                status_color = "#9C27B0"
            else:
                status_icon = "✅"
                status_color = "#4CAF50"

            # Format timestamp
            try:
                ts = datetime.fromisoformat(i['timestamp'])
                time_str = ts.strftime("%b %d, %H:%M")
                time_ago = datetime.now() - ts
                if time_ago.days > 0:
                    ago_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    ago_str = f"{time_ago.seconds // 3600}h ago"
                elif time_ago.seconds > 60:
                    ago_str = f"{time_ago.seconds // 60}m ago"
                else:
                    ago_str = "just now"
            except:
                time_str = i['timestamp'][:16]
                ago_str = ""

            # Create expandable card
            header = f"{status_icon} **{i['agent_name']}** · {time_str} ({ago_str})"
            with st.expander(header):
                # Metadata row
                meta_cols = st.columns([1, 1, 1, 1])
                with meta_cols[0]:
                    duration_s = i['duration_ms'] / 1000
                    st.caption(f"⏱️ {duration_s:.2f}s")
                with meta_cols[1]:
                    st.caption(f"🎟️ {i['tokens_out']} tokens")
                with meta_cols[2]:
                    tool_count = len(i.get('tool_calls', []))
                    st.caption(f"🔧 {tool_count} tools")
                with meta_cols[3]:
                    if i.get('error'):
                        st.caption("❌ Error")
                    else:
                        st.caption("✅ Success")

                # Messages
                st.markdown("**👤 User:**")
                st.markdown(f"> {i['user_message']}")

                st.markdown("**🤖 Assistant:**")
                assistant_msg = i['assistant_message']
                if len(assistant_msg) > 500:
                    st.markdown(f"> {assistant_msg[:500]}...")
                    with st.popover("Show full response"):
                        st.markdown(assistant_msg)
                else:
                    st.markdown(f"> {assistant_msg}")

                # Error if any
                if i.get("error"):
                    st.error(f"**Error:** {i['error']}")

                # Tool calls
                if i.get("tool_calls"):
                    st.markdown("**🔧 Tool Calls:**")
                    for tc in i["tool_calls"]:
                        success_icon = "✅" if tc.get("success", True) else "❌"
                        with st.container():
                            st.code(f"{success_icon} {tc.get('name')}\n"
                                   f"   Args: {json.dumps(tc.get('arguments', {}), indent=2)[:200]}")

                # Todos
                if i.get("todos"):
                    st.markdown("**📝 Todos:**")
                    for todo in i["todos"]:
                        status_icons = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}
                        icon = status_icons.get(todo.get("status"), "⬜")
                        st.markdown(f"  {icon} {todo.get('content')}")

    st.divider()

    # ========== ACTIONS ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh", key="refresh_history", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("📥 Export JSON", key="export_history", use_container_width=True):
            all_interactions = store.get_interactions(limit=1000)
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "statistics": stats,
                "interactions": all_interactions,
            }
            st.download_button(
                "📥 Download",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"deepagents_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_history"
            )

    with col3:
        def clear_all_history():
            store.clear_history()
            add_debug_event("HISTORY", "All history cleared")
            st.toast("🗑️ All history cleared", icon="✅")

        st.button("🗑️ Clear All History", key="clear_history",
                 use_container_width=True, on_click=clear_all_history, type="secondary")


# ============================================================================
# Top Bar & Control Panel (replaces sidebar)
# ============================================================================

def render_top_bar():
    """Render the comprehensive control panel with connection, model, and session controls."""

    # ===== HEADER ROW =====
    st.markdown("""
    <div class="app-header" style="margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1 style="margin: 0; font-size: 1.5rem;">🦜 LangChain Agent Explorer</h1>
                <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.85rem;">Build, test, and debug AI agents with Ollama</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== CONTROL PANEL =====
    # Determine states
    if st.session_state.ollama_connected:
        conn_icon, conn_bg, conn_color = "🟢", "#ecfdf5", "#047857"
    elif st.session_state.ollama_connected is None:
        conn_icon, conn_bg, conn_color = "⚪", "#f8fafc", "#64748b"
    else:
        conn_icon, conn_bg, conn_color = "🔴", "#fef2f2", "#dc2626"

    active_id = st.session_state.active_agent_id
    if active_id and active_id in st.session_state.saved_agents:
        agent = st.session_state.saved_agents[active_id]
        agent_name = agent.get("name", "Unknown")
        agent_model = agent.get("model", "Unknown")
        mw_count = len(agent.get("middleware") or [])
        tool_count = len(agent.get("tools") or [])
    else:
        agent_name = None
        agent_model = st.session_state.ollama_config.model
        mw_count = len(st.session_state.selected_middleware)
        tool_count = len(st.session_state.selected_tools)

    exec_mgr = get_execution_manager()
    active_tasks = exec_mgr.get_active_count()

    # Control panel container
    st.markdown("""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                padding: 0.875rem 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    """, unsafe_allow_html=True)

    # Four column layout for controls
    col_conn, col_model, col_agent, col_session = st.columns([2.5, 2.5, 3, 2])

    # ===== CONNECTION SECTION =====
    with col_conn:
        st.markdown(f"""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
                    color: #64748b; margin-bottom: 0.4rem; font-weight: 600;">🔌 Connection</div>
        """, unsafe_allow_html=True)

        # Connection status badge
        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.65rem;
                    background: {conn_bg}; border-radius: 6px; font-size: 0.8rem; margin-bottom: 0.5rem;">
            <span>{conn_icon}</span>
            <span style="color: {conn_color}; font-weight: 500;">
                {'Connected' if st.session_state.ollama_connected else 'Disconnected' if st.session_state.ollama_connected is False else 'Not Checked'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        btn_row = st.columns([1, 1, 1])
        with btn_row[0]:
            def check_connection():
                try:
                    result = test_ollama_connection(st.session_state.ollama_config.base_url, timeout=3.0)
                    st.session_state.ollama_connected = result.success
                    if result.success:
                        client = OllamaClient(st.session_state.ollama_config)
                        st.session_state.available_models = client.list_models()
                        st.toast(f"✅ Connected ({result.latency_ms:.0f}ms)", icon="🟢")
                    else:
                        st.toast("❌ Connection failed", icon="🔴")
                except Exception as e:
                    st.session_state.ollama_connected = False
                    st.toast(f"❌ {e}", icon="🔴")
            st.button("🔄 Check", key="ctrl_check", on_click=check_connection, use_container_width=True)

        with btn_row[1]:
            def load_model():
                client = OllamaClient(st.session_state.ollama_config)
                if client.warmup_model():
                    st.session_state.model_loaded = True
                    st.session_state.agent_instance = None
                    st.toast("✅ Model loaded!", icon="🔥")
                    st.rerun()
                else:
                    st.toast("❌ Load failed", icon="⚠️")
            st.button("🔥 Load", key="ctrl_load", on_click=load_model,
                     disabled=not st.session_state.ollama_connected or st.session_state.get("model_loaded", False),
                     use_container_width=True)

        with btn_row[2]:
            # Settings popover
            with st.popover("⚙️", use_container_width=True):
                st.markdown("**Ollama Settings**")
                new_url = st.text_input("Base URL", value=st.session_state.ollama_config.base_url,
                                       key="ctrl_url", label_visibility="collapsed")
                if new_url != st.session_state.ollama_config.base_url:
                    st.session_state.ollama_config.base_url = new_url
                    st.session_state.ollama_connected = None
                    st.session_state.model_loaded = False

    # ===== MODEL INFO SECTION (Read-only, model set during agent creation) =====
    with col_model:
        st.markdown(f"""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
                    color: #64748b; margin-bottom: 0.4rem; font-weight: 600;">🧠 Model</div>
        """, unsafe_allow_html=True)

        # Display current model (read-only - set during agent creation)
        current_model = st.session_state.ollama_config.model
        model_ready = st.session_state.get("model_loaded", False)

        st.markdown(f"""
        <div style="background: {'#ecfdf5' if model_ready else '#f8fafc'}; border: 1px solid {'#6ee7b7' if model_ready else '#e2e8f0'};
                    border-radius: 8px; padding: 0.5rem 0.75rem;">
            <div style="font-weight: 600; color: #1e293b; font-size: 0.85rem;">{current_model}</div>
            <div style="font-size: 0.7rem; color: {'#047857' if model_ready else '#64748b'}; margin-top: 0.2rem;">
                {'✓ Ready' if model_ready else '⚡ Click Load to warm up'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick context info
        ctx_window = st.session_state.ollama_config.num_ctx
        temp = st.session_state.ollama_config.temperature
        st.markdown(f"""
        <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.3rem;">
            ctx: {ctx_window:,} · temp: {temp}
        </div>
        """, unsafe_allow_html=True)

    # ===== ACTIVE AGENT SECTION =====
    with col_agent:
        st.markdown(f"""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
                    color: #64748b; margin-bottom: 0.4rem; font-weight: 600;">🤖 Active Agent</div>
        """, unsafe_allow_html=True)

        if agent_name:
            # Agent info with quick stats
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                <span style="background: linear-gradient(135deg, #10b981, #059669); color: white;
                            padding: 0.2rem 0.5rem; border-radius: 5px; font-size: 0.75rem; font-weight: 600;">
                    {agent_name}
                </span>
                <span style="font-size: 0.7rem; color: #64748b;">{mw_count} mw · {tool_count} tools</span>
            </div>
            """, unsafe_allow_html=True)

            # Quick agent actions
            agent_btns = st.columns([1, 1, 1])
            with agent_btns[0]:
                # Agent switcher popover
                with st.popover("🔄", use_container_width=True):
                    st.markdown("**Switch Agent**")
                    for aid, adata in st.session_state.saved_agents.items():
                        is_current = aid == active_id
                        btn_type = "secondary" if is_current else "primary"
                        if st.button(f"{'✓ ' if is_current else ''}{adata['name']}", key=f"switch_{aid}",
                                    disabled=is_current, use_container_width=True):
                            # Activate this agent
                            st.session_state.active_agent_id = aid
                            ag = st.session_state.saved_agents[aid]
                            st.session_state.agent_config.name = ag["name"]
                            st.session_state.agent_config.description = ag.get("description", "")
                            ag_mw = ag.get("middleware") or []
                            st.session_state.selected_middleware = [
                                MiddlewareType(m) for m in ag_mw if m in [mt.value for mt in MiddlewareType]
                            ]
                            st.session_state.selected_tools = ag.get("tools") or []
                            st.session_state.ollama_config.model = ag.get("model", "llama3.2:latest")
                            st.session_state.agent_instance = None
                            st.toast(f"🤖 Switched to {ag['name']}", icon="✅")
                            st.rerun()

            with agent_btns[1]:
                def deactivate_agent():
                    st.session_state.active_agent_id = None
                    st.session_state.agent_instance = None
                    st.toast("Agent deactivated", icon="👋")
                st.button("✖️", key="deactivate_agent", on_click=deactivate_agent, use_container_width=True,
                         help="Deactivate agent")

            with agent_btns[2]:
                # Test status
                if active_id:
                    test_passed = st.session_state.saved_agents[active_id].get("test_passed")
                    tested = st.session_state.saved_agents[active_id].get("tested")
                    if test_passed:
                        st.markdown("""<div style="text-align: center; color: #10b981;">✅</div>""",
                                   unsafe_allow_html=True)
                    elif tested:
                        st.markdown("""<div style="text-align: center; color: #ef4444;">❌</div>""",
                                   unsafe_allow_html=True)
                    else:
                        st.markdown("""<div style="text-align: center; color: #f59e0b;">🧪</div>""",
                                   unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #fef3c7; border: 1px solid #fde68a; border-radius: 8px;
                        padding: 0.5rem 0.75rem; text-align: center;">
                <span style="color: #92400e; font-size: 0.85rem;">⚠️ No agent selected</span>
                <br><span style="color: #b45309; font-size: 0.7rem;">Go to Agent Studio to create one</span>
            </div>
            """, unsafe_allow_html=True)

    # ===== SESSION SECTION =====
    with col_session:
        st.markdown(f"""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
                    color: #64748b; margin-bottom: 0.4rem; font-weight: 600;">📋 Session</div>
        """, unsafe_allow_html=True)

        # Session info
        session_id = st.session_state.session_id[:8]
        thread_id = st.session_state.thread_id[:8]
        msg_count = len(st.session_state.messages)

        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.4rem;">
            <code>{session_id}</code> · {msg_count} msgs
            {'<span style="color: #3b82f6;">🔄 ' + str(active_tasks) + '</span>' if active_tasks > 0 else ''}
        </div>
        """, unsafe_allow_html=True)

        session_btns = st.columns([1, 1])
        with session_btns[0]:
            def clear_session():
                st.session_state.messages = []
                st.session_state.debug_events = []
                st.session_state.current_steps = []
                st.session_state.tool_calls = []
                st.session_state.todos = []
                st.session_state.current_phase = "idle"
                save_current_conversation()
                st.toast("🗑️ Chat cleared", icon="✅")
            st.button("🗑️ Clear", key="ctrl_clear", on_click=clear_session, use_container_width=True)

        with session_btns[1]:
            def new_session():
                save_current_conversation()
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.agent_instance = None
                st.session_state.current_steps = []
                st.session_state.tool_calls = []
                st.session_state.todos = []
                st.session_state.active_conversation_id = None
                st.toast("🔄 New session", icon="✅")
            st.button("🔄 New", key="ctrl_new", on_click=new_session, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ===== QUICK STATS BAR =====
    if st.session_state.debug_enabled:
        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
        with stats_col1:
            st.markdown(f"""<div style="font-size: 0.75rem; color: #64748b; text-align: center;">
                📝 <strong>{len(st.session_state.messages)}</strong> messages</div>""", unsafe_allow_html=True)
        with stats_col2:
            st.markdown(f"""<div style="font-size: 0.75rem; color: #64748b; text-align: center;">
                🔧 <strong>{len(st.session_state.tool_calls)}</strong> tool calls</div>""", unsafe_allow_html=True)
        with stats_col3:
            st.markdown(f"""<div style="font-size: 0.75rem; color: #64748b; text-align: center;">
                📋 <strong>{len(st.session_state.current_steps)}</strong> steps</div>""", unsafe_allow_html=True)
        with stats_col4:
            st.markdown(f"""<div style="font-size: 0.75rem; color: #64748b; text-align: center;">
                ✅ <strong>{len(st.session_state.todos)}</strong> todos</div>""", unsafe_allow_html=True)
        with stats_col5:
            st.markdown(f"""<div style="font-size: 0.75rem; color: #64748b; text-align: center;">
                ⏳ <strong>{len(st.session_state.pending_approvals)}</strong> pending</div>""", unsafe_allow_html=True)

    # Debug toggle (small, at the side)
    debug_col1, debug_col2 = st.columns([6, 1])
    with debug_col2:
        st.session_state.debug_enabled = st.checkbox("🔍", value=st.session_state.debug_enabled,
                                                    key="debug_toggle_ctrl", help="Toggle debug mode")


# ============================================================================
# Unified Chat Workspace (Chat + Panels)
# ============================================================================

def render_chat_workspace():
    """Render the unified chat workspace with embedded panels."""
    # Check if an agent is active
    active_id = st.session_state.active_agent_id
    if not active_id or active_id not in st.session_state.saved_agents:
        # No agent - show guidance
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    border-radius: 16px; border: 1px solid #e2e8f0;">
            <span style="font-size: 4rem; opacity: 0.4;">🤖</span>
            <h2 style="color: #64748b; margin: 1rem 0;">No Agent Selected</h2>
            <p style="color: #94a3b8; max-width: 400px; margin: 0 auto;">
                Go to the <strong>Agent Studio</strong> tab to create and activate an agent first.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Get active agent info
    active_agent = st.session_state.saved_agents[active_id]
    exec_mgr = get_execution_manager()

    # Main layout: Chat (left) | Panels (right)
    chat_col, panel_col = st.columns([3, 2])

    with chat_col:
        # Chat header with conversation selector and stats
        total_tokens = sum(msg.get("tokens_out", 0) or 0 for msg in st.session_state.messages if msg.get("role") == "assistant")
        last_response_time = None
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "assistant" and msg.get("duration_ms"):
                last_response_time = msg.get("duration_ms") / 1000
                break

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white;
                    padding: 0.75rem 1rem; border-radius: 10px 10px 0 0; margin-bottom: 0;
                    display: flex; justify-content: space-between; align-items: center;">
            <strong>💬 Chat</strong>
            <div style="font-size: 0.75rem; opacity: 0.9;">
                {f'🎟️ {total_tokens} tokens' if total_tokens > 0 else ''}
                {f' · ⏱️ {last_response_time:.1f}s' if last_response_time else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Conversation controls row
        conv_row = st.columns([2.5, 1, 1, 0.5])
        with conv_row[0]:
            active_conv_id = st.session_state.active_conversation_id
            if active_conv_id and active_conv_id in st.session_state.conversations:
                conv = st.session_state.conversations[active_conv_id]
                conv_name = st.text_input("", value=conv.get("name", "Conversation"),
                                         key="conv_name_input", label_visibility="collapsed")
                if conv_name != conv.get("name"):
                    rename_conversation(active_conv_id, conv_name)
            else:
                st.caption("No conversation selected")

        with conv_row[1]:
            if st.button("➕ New", key="new_conv_workspace", use_container_width=True):
                conv_id = create_new_conversation()
                st.session_state.active_conversation_id = conv_id
                st.session_state.messages = []
                st.session_state.thread_id = st.session_state.conversations[conv_id]["thread_id"]
                st.toast("✨ New conversation!", icon="💬")
                st.rerun()

        with conv_row[2]:
            # Conversation dropdown
            conversations = st.session_state.conversations
            if conversations:
                conv_ids = list(conversations.keys())
                conv_names = [conversations[cid].get("name", cid)[:20] for cid in conv_ids]
                current_idx = conv_ids.index(active_conv_id) if active_conv_id in conv_ids else 0
                selected_idx = st.selectbox("", range(len(conv_ids)),
                                           format_func=lambda i: conv_names[i],
                                           index=current_idx, key="conv_selector",
                                           label_visibility="collapsed")
                if conv_ids[selected_idx] != active_conv_id:
                    switch_conversation(conv_ids[selected_idx])
                    st.rerun()

        with conv_row[3]:
            # FEATURE 1: Copy last response button
            if st.session_state.messages:
                last_assistant_msg = None
                for msg in reversed(st.session_state.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        last_assistant_msg = msg.get("content")
                        break
                if last_assistant_msg:
                    # Use popover to show copy confirmation
                    with st.popover("📋", help="Copy last response"):
                        st.code(last_assistant_msg[:500], language=None)
                        st.caption("Select and copy the text above")

        # FEATURE 2: Quick prompt templates
        with st.expander("⚡ Quick Prompts", expanded=False):
            quick_prompts = [
                ("📝 Summarize", "Please summarize our conversation so far."),
                ("🔍 Explain", "Can you explain that in more detail?"),
                ("💡 Examples", "Can you give me some examples?"),
                ("🔄 Rephrase", "Can you rephrase that in simpler terms?"),
                ("📋 List", "Can you list the key points?"),
                ("🐛 Debug", "Help me debug this issue."),
            ]
            qp_cols = st.columns(3)
            for i, (label, prompt_text) in enumerate(quick_prompts):
                with qp_cols[i % 3]:
                    if st.button(label, key=f"qp_{i}", use_container_width=True):
                        # Add as user message and trigger response
                        st.session_state.messages.append({"role": "user", "content": prompt_text})
                        save_current_conversation()
                        st.session_state["_pending_quick_prompt"] = prompt_text
                        st.rerun()

        # FEATURE 3: Conversation search
        with st.expander("🔍 Search Conversation", expanded=False):
            search_query = st.text_input("Search messages", key="conv_search_query",
                                        placeholder="Type to search...", label_visibility="collapsed")
            if search_query and st.session_state.messages:
                matches = []
                for i, msg in enumerate(st.session_state.messages):
                    content = msg.get("content", "")
                    if search_query.lower() in content.lower():
                        matches.append((i, msg))
                if matches:
                    st.caption(f"Found {len(matches)} matches")
                    for idx, msg in matches[:5]:  # Show first 5 matches
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        content_preview = msg.get("content", "")[:100]
                        # Highlight search term
                        lower_preview = content_preview.lower()
                        lower_query = search_query.lower()
                        if lower_query in lower_preview:
                            start = lower_preview.find(lower_query)
                            highlighted = f"{content_preview[:start]}**{content_preview[start:start+len(search_query)]}**{content_preview[start+len(search_query):]}"
                        else:
                            highlighted = content_preview
                        st.markdown(f"{role_icon} #{idx+1}: {highlighted}...")
                else:
                    st.caption("No matches found")

        # Chat messages
        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg.get("error"):
                        st.error(msg["error"])
                    else:
                        st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("duration_ms"):
                        parts = []
                        if msg.get("tokens_per_second"):
                            parts.append(f"{msg['tokens_per_second']:.1f} tok/s")
                        if msg.get("tool_calls"):
                            parts.append(f"{len(msg['tool_calls'])} tools")
                        if parts:
                            st.caption(" | ".join(parts))

        # Chat input
        col_input, col_mode = st.columns([5, 1])
        with col_mode:
            send_mode = st.selectbox("Mode", ["Blocking", "Background"],
                                    key="workspace_send_mode", label_visibility="collapsed")
        with col_input:
            if prompt := st.chat_input("Message...", key="workspace_chat_input"):
                if not st.session_state.active_conversation_id:
                    conv_id = create_new_conversation()
                    st.session_state.active_conversation_id = conv_id

                if st.session_state.ollama_connected is False:
                    st.error("❌ Ollama not connected")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    save_current_conversation()

                    if send_mode == "Background":
                        task_id = send_background_prompt(prompt)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⏳ Task {task_id} running in background..."
                        })
                        st.toast(f"🔄 Background task {task_id} started", icon="⏳")
                    else:
                        with st.chat_message("assistant"):
                            response_ph = st.empty()
                            status_ph = st.empty()
                            response = get_streaming_response(prompt, response_ph, status_ph)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response.get("message", ""),
                                "duration_ms": response.get("duration_ms"),
                                "tokens_per_second": response.get("tokens_per_second"),
                                "tool_calls": response.get("tool_calls"),
                                "error": response.get("error"),
                            })
                        save_current_conversation()
                    st.rerun()

    with panel_col:
        # Panel tabs
        panel_tabs = st.tabs(["📋 Steps", "🔧 Tools", "📝 Todos", "👤 Approvals", "🔄 Tasks"])

        with panel_tabs[0]:
            # Steps panel (compact)
            if st.session_state.current_steps:
                for step in st.session_state.current_steps[-5:]:
                    step_type = step["type"]
                    icons = {"thinking": "🧠", "tool": "🔧", "response": "💬", "approval": "⏳"}
                    st.markdown(f"**{icons.get(step_type, '⚪')} {step_type.title()}** - {step['timestamp'][11:19]}")
                    st.caption(str(step['content'])[:100])
            else:
                st.info("No steps yet")
            if st.button("🗑️ Clear", key="clear_steps_panel"):
                st.session_state.current_steps = []

        with panel_tabs[1]:
            # Tools panel (compact)
            if st.session_state.tool_calls:
                for call in st.session_state.tool_calls[-5:]:
                    status_icon = {"pending": "⏳", "completed": "✅", "failed": "❌"}.get(call["status"], "⚪")
                    st.markdown(f"**{status_icon} {call['tool_name']}**")
                    st.code(json.dumps(call["args"], indent=1)[:150], language="json")
            else:
                st.info("No tool calls yet")

        with panel_tabs[2]:
            # Todos panel (compact)
            if st.session_state.todos:
                for todo in st.session_state.todos:
                    icons = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
                    st.markdown(f"{icons.get(todo.get('status'), '⬜')} {todo.get('content', 'Task')[:50]}")
            else:
                st.info("No todos yet")

        with panel_tabs[3]:
            # Approvals panel (compact)
            pending = st.session_state.pending_approvals
            if pending:
                for approval in pending:
                    st.warning(f"⚠️ **{approval.get('tool_name')}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅", key=f"approve_panel_{approval['id']}"):
                            approval["status"] = "approved"
                            st.session_state.approval_history.append(approval)
                            st.session_state.pending_approvals.remove(approval)
                            st.rerun()
                    with col2:
                        if st.button("❌", key=f"reject_panel_{approval['id']}"):
                            approval["status"] = "rejected"
                            st.session_state.approval_history.append(approval)
                            st.session_state.pending_approvals.remove(approval)
                            st.rerun()
            else:
                st.info("No pending approvals")

        with panel_tabs[4]:
            # Tasks panel (compact)
            active_tasks = exec_mgr.get_active_tasks_info()
            if active_tasks:
                for task in active_tasks:
                    st.markdown(f"🔵 **{task['task_id']}** - {task.get('current_step', 'running')}")
                    st.progress(min(1.0, task.get('elapsed_ms', 0) / 30000))
            completed = st.session_state.task_results[-3:]
            if completed:
                st.markdown("**Recent:**")
                for r in reversed(completed):
                    icon = "✅" if r.get("status") == "completed" else "❌"
                    st.caption(f"{icon} {r.get('task_id')} - {r.get('duration_ms', 0)/1000:.1f}s")
            if not active_tasks and not completed:
                st.info("No tasks")


# ============================================================================
# Agent Studio (Builder + Test unified)
# ============================================================================

def render_agent_studio():
    """Render the unified agent studio with builder and test side by side."""
    # Layout: Builder (left) | Saved Agents + Test (right)
    col_builder, col_agents = st.columns([3, 2])

    with col_builder:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white;
                    padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <strong>🔧 Agent Builder</strong>
            <span style="opacity: 0.8; font-size: 0.85rem; margin-left: 0.5rem;">Configure your agent</span>
        </div>
        """, unsafe_allow_html=True)

        builder = st.session_state.builder_agent_config

        # Basic Info
        col_name, col_model = st.columns(2)
        with col_name:
            builder["name"] = st.text_input("Agent Name", value=builder.get("name", "New Agent"),
                                           key="studio_name")
        with col_model:
            models = st.session_state.available_models
            model_names = [m.get("name", "") for m in models] if models else []
            curr_model = builder.get("model", "llama3.2:latest")
            if curr_model not in model_names:
                model_names.insert(0, curr_model)
            builder["model"] = st.selectbox("Model", model_names,
                                           index=model_names.index(curr_model) if curr_model in model_names else 0,
                                           key="studio_model")

        builder["description"] = st.text_area("Description", value=builder.get("description", ""),
                                             height=60, key="studio_desc")

        # Model settings
        col_temp, col_ctx = st.columns(2)
        with col_temp:
            builder["temperature"] = st.slider("Temperature", 0.0, 2.0,
                                              builder.get("temperature", 0.7), 0.1, key="studio_temp")
        with col_ctx:
            builder["num_ctx"] = st.select_slider("Context Window",
                                                 options=[2048, 4096, 8192, 16384, 32768],
                                                 value=builder.get("num_ctx", 8192), key="studio_ctx")

        # Middleware Selection - Use multiselect to avoid duplicates
        st.markdown("**🔧 Middleware**")
        configs = get_middleware_configs()
        sorted_configs = sorted(configs, key=lambda x: x.priority)

        # Get current middleware as list of MiddlewareType
        current_middleware = builder.get("middleware") or []
        if not isinstance(current_middleware, list):
            current_middleware = []
        # Ensure no duplicates by converting to set and back
        current_middleware = list(dict.fromkeys(current_middleware))

        # Use multiselect for clean selection without duplicates
        mw_options = [cfg.middleware_type for cfg in sorted_configs]
        mw_labels = {cfg.middleware_type: f"{cfg.name}" for cfg in sorted_configs}
        mw_descriptions = {cfg.middleware_type: cfg.description for cfg in sorted_configs}

        selected_mw = st.multiselect(
            "Select Middleware",
            options=mw_options,
            default=[m for m in current_middleware if m in mw_options],
            format_func=lambda x: mw_labels.get(x, str(x)),
            key="studio_mw_select",
            help="Select middleware components to enhance agent capabilities",
            label_visibility="collapsed"
        )
        # Store as unique list (no duplicates possible with multiselect)
        builder["middleware"] = list(dict.fromkeys(selected_mw))

        # Tools Selection
        st.markdown("**🛠️ Tools**")
        tool_registry = get_tool_registry()
        available_tools = list(tool_registry.get_tool_info().keys())
        current_tools = builder.get("tools") or []
        builder["tools"] = st.multiselect("Select Tools", options=available_tools,
                                         default=[t for t in current_tools if t in available_tools],
                                         key="studio_tools", label_visibility="collapsed")

        # System Prompt (collapsible)
        with st.expander("📜 System Prompt (optional)"):
            builder["system_prompt"] = st.text_area("Custom instructions",
                                                   value=builder.get("system_prompt", ""),
                                                   height=100, key="studio_prompt",
                                                   label_visibility="collapsed")

        # Save Button
        st.markdown("---")
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            save_name = st.text_input("Save as", value=builder.get("name", "New Agent"),
                                     key="studio_save_name", label_visibility="collapsed")
        with save_col2:
            def save_agent_studio():
                agent_id = str(uuid.uuid4())[:8]
                agent_data = {
                    "id": agent_id,
                    "name": save_name,
                    "description": builder.get("description", ""),
                    "system_prompt": builder.get("system_prompt", ""),
                    "middleware": [m.value for m in builder.get("middleware", [])],
                    "tools": builder.get("tools", []),
                    "model": builder.get("model", "llama3.2:latest"),
                    "temperature": builder.get("temperature", 0.7),
                    "num_ctx": builder.get("num_ctx", 8192),
                    "created_at": datetime.now().isoformat(),
                    "tested": False,
                    "test_passed": False,
                }
                st.session_state.saved_agents[agent_id] = agent_data
                st.toast(f"✅ Agent '{save_name}' saved!", icon="💾")
                st.rerun()

            st.button("💾 Save", key="studio_save_btn", on_click=save_agent_studio,
                     use_container_width=True, type="primary")

    with col_agents:
        # Header with import button
        header_col, import_col = st.columns([3, 1])
        with header_col:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;
                        padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <strong>📚 Saved Agents</strong>
            </div>
            """, unsafe_allow_html=True)
        with import_col:
            uploaded = st.file_uploader("Import", type="json", key="import_agent_file",
                                       label_visibility="collapsed")
            if uploaded:
                try:
                    imported = json.load(uploaded)
                    # Generate new ID for imported agent
                    new_id = str(uuid.uuid4())[:8]
                    imported["id"] = new_id
                    imported["name"] = imported.get("name", "Imported Agent") + " (imported)"
                    imported["created_at"] = datetime.now().isoformat()
                    imported["tested"] = False
                    imported["test_passed"] = False
                    st.session_state.saved_agents[new_id] = imported
                    st.toast(f"📥 Imported '{imported['name']}'!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

        saved = st.session_state.saved_agents
        if not saved:
            st.info("No saved agents. Configure and save one to get started.")
        else:
            for agent_id, agent_data in saved.items():
                is_active = st.session_state.active_agent_id == agent_id
                test_icon = "✅" if agent_data.get("test_passed") else ("🧪" if agent_data.get("tested") else "⬜")

                border_color = "#22c55e" if is_active else "#e2e8f0"
                bg_color = "#f0fdf4" if is_active else "white"

                st.markdown(f"""
                <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 10px;
                            padding: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{agent_data['name']}</strong>
                        <span>{test_icon}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b;">
                        {agent_data.get('model', '?')} · {len(agent_data.get('middleware') or [])} mw · {len(agent_data.get('tools') or [])} tools
                    </div>
                </div>
                """, unsafe_allow_html=True)

                btn_cols = st.columns(5)
                with btn_cols[0]:
                    def activate(aid=agent_id):
                        st.session_state.active_agent_id = aid
                        agent = st.session_state.saved_agents[aid]
                        st.session_state.agent_config.name = agent["name"]
                        st.session_state.agent_config.description = agent.get("description", "")
                        st.session_state.agent_config.system_prompt = agent.get("system_prompt")
                        st.session_state.ollama_config.model = agent.get("model", "llama3.2:latest")
                        st.session_state.ollama_config.temperature = agent.get("temperature", 0.7)
                        st.session_state.ollama_config.num_ctx = agent.get("num_ctx", 8192)
                        agent_middleware = agent.get("middleware") or []
                        st.session_state.selected_middleware = [
                            MiddlewareType(m) for m in agent_middleware if m in [mt.value for mt in MiddlewareType]
                        ]
                        st.session_state.selected_tools = agent.get("tools") or []
                        st.session_state.agent_instance = None
                        st.session_state.messages = []
                        st.toast(f"🤖 Agent '{agent['name']}' activated!", icon="✅")

                    st.button("⚡" if not is_active else "✓", key=f"activate_studio_{agent_id}",
                             on_click=activate, disabled=is_active, use_container_width=True,
                             help="Activate agent")

                with btn_cols[1]:
                    def load_to_builder(aid=agent_id):
                        agent = st.session_state.saved_agents[aid]
                        agent_middleware = agent.get("middleware") or []
                        st.session_state.builder_agent_config = {
                            "name": agent["name"],
                            "description": agent.get("description", ""),
                            "system_prompt": agent.get("system_prompt", ""),
                            "middleware": [MiddlewareType(m) for m in agent_middleware if m in [mt.value for mt in MiddlewareType]],
                            "tools": agent.get("tools") or [],
                            "model": agent.get("model", "llama3.2:latest"),
                            "temperature": agent.get("temperature", 0.7),
                            "num_ctx": agent.get("num_ctx", 8192),
                        }
                    st.button("📝", key=f"edit_studio_{agent_id}", on_click=load_to_builder,
                             use_container_width=True, help="Edit in builder")

                with btn_cols[2]:
                    def test_agent(aid=agent_id):
                        # Activate first, then test
                        st.session_state.active_agent_id = aid
                        agent = st.session_state.saved_agents[aid]
                        agent_middleware = agent.get("middleware") or []
                        st.session_state.selected_middleware = [
                            MiddlewareType(m) for m in agent_middleware if m in [mt.value for mt in MiddlewareType]
                        ]
                        st.session_state.selected_tools = agent.get("tools") or []
                        st.session_state.agent_instance = None
                        st.session_state.ollama_config.model = agent.get("model", "llama3.2:latest")
                        st.session_state["testing_agent"] = aid

                    st.button("🧪", key=f"test_studio_{agent_id}", on_click=test_agent,
                             use_container_width=True, help="Test agent")

                with btn_cols[3]:
                    # Export agent as JSON
                    export_data = json.dumps(agent_data, indent=2, default=str)
                    st.download_button(
                        "📤",
                        data=export_data,
                        file_name=f"{agent_data['name'].replace(' ', '_')}_agent.json",
                        mime="application/json",
                        key=f"export_studio_{agent_id}",
                        use_container_width=True,
                        help="Export agent"
                    )

                with btn_cols[4]:
                    def delete_agent(aid=agent_id):
                        name = st.session_state.saved_agents[aid].get("name", "Agent")
                        del st.session_state.saved_agents[aid]
                        if st.session_state.active_agent_id == aid:
                            st.session_state.active_agent_id = None
                            st.session_state.agent_instance = None
                        st.toast(f"🗑️ Deleted '{name}'", icon="🗑️")

                    st.button("🗑️", key=f"delete_studio_{agent_id}", on_click=delete_agent, use_container_width=True)

        # Test Panel (shown when testing)
        if st.session_state.get("testing_agent"):
            st.markdown("---")
            st.markdown("### 🧪 Test Agent")
            testing_id = st.session_state["testing_agent"]
            if testing_id in st.session_state.saved_agents:
                agent_data = st.session_state.saved_agents[testing_id]
                st.info(f"Testing: **{agent_data['name']}**")

                test_prompt = st.text_input("Test prompt", value="Hello! What can you do?", key="studio_test_prompt")

                if st.button("▶️ Run Test", key="run_studio_test", type="primary", use_container_width=True):
                    with st.spinner("Running test..."):
                        try:
                            response_ph = st.empty()
                            status_ph = st.empty()
                            response = get_streaming_response(test_prompt, response_ph, status_ph)

                            if response.get("error"):
                                st.error(f"❌ Test failed: {response['error']}")
                                agent_data["tested"] = True
                                agent_data["test_passed"] = False
                            else:
                                st.success(f"✅ Test passed! ({response.get('duration_ms', 0)/1000:.2f}s)")
                                agent_data["tested"] = True
                                agent_data["test_passed"] = True
                                st.toast("✅ Test passed!", icon="🎉")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            agent_data["tested"] = True
                            agent_data["test_passed"] = False

                if st.button("✖️ Close Test", key="close_test"):
                    del st.session_state["testing_agent"]
                    st.rerun()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    init_session_state()

    # Auto-check Ollama connection on first load
    if st.session_state.ollama_connected is None:
        try:
            result = test_ollama_connection(
                st.session_state.ollama_config.base_url,
                timeout=2.0
            )
            st.session_state.ollama_connected = result.success
            if result.success:
                client = OllamaClient(st.session_state.ollama_config)
                st.session_state.available_models = client.list_models()
                add_debug_event("STARTUP", f"Connected to Ollama ({result.latency_ms:.0f}ms)")
            else:
                add_debug_event("STARTUP", f"Ollama not available: {result.error}", "error")
        except Exception as e:
            st.session_state.ollama_connected = False
            add_debug_event("STARTUP", f"Connection check failed: {e}", "error")

    # Top bar (replaces sidebar)
    render_top_bar()

    # Main tabs - simplified to two main views
    tabs = st.tabs([
        "🎨 Agent Studio",
        "💬 Chat Workspace",
        "⏰ Schedule",
        "📊 History & Analytics"
    ])

    with tabs[0]:
        render_agent_studio()

    with tabs[1]:
        render_chat_workspace()

    with tabs[2]:
        scheduled_prompts_fragment()

    with tabs[3]:
        history_fragment()

    # Compact footer
    st.markdown("---")
    cfg = st.session_state.ollama_config
    middleware_count = len(st.session_state.selected_middleware)
    exec_mgr = get_execution_manager()
    active_tasks = exec_mgr.get_active_count()

    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; padding: 0.4rem;
                font-size: 0.75rem; color: #94a3b8;">
        <span>🧠 {cfg.model}</span>
        <span>🔧 {middleware_count} middleware</span>
        <span>{'🔄 ' + str(active_tasks) + ' running' if active_tasks > 0 else '💤 Idle'}</span>
        <span style="color: #a855f7;">Powered by LangChain</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
