"""
Observability Engine for DeepAgents

This module provides comprehensive execution tracing, metrics collection,
and performance monitoring for agent interactions.
"""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
import threading


class EventType(str, Enum):
    """Types of events that can be traced."""
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MIDDLEWARE = "middleware"
    ERROR = "error"
    USER_INPUT = "user_input"
    AGENT_OUTPUT = "agent_output"
    SKILL_APPLIED = "skill_applied"
    SECRET_INJECTED = "secret_injected"


class MetricType(str, Enum):
    """Types of metrics to collect."""
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    ERROR_RATE = "error_rate"
    TOOL_USAGE = "tool_usage"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"


@dataclass
class ExecutionTrace:
    """A single trace event in the execution timeline."""
    id: str
    conversation_id: str
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    parent_id: Optional[str] = None
    status: str = "pending"  # pending, completed, error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "parent_id": self.parent_id,
            "status": self.status,
        }


@dataclass
class MetricPoint:
    """A single metric data point."""
    metric_type: MetricType
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value if isinstance(self.metric_type, MetricType) else self.metric_type,
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
        }


class ExecutionTracer:
    """
    Captures detailed execution traces for observability.

    Provides hierarchical tracing of all agent operations including:
    - LLM requests and responses
    - Tool calls and results
    - Middleware processing
    - Errors and exceptions
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._active_spans: Dict[str, ExecutionTrace] = {}
        self._conversation_traces: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Ensure the traces table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_traces (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                data TEXT,
                duration_ms REAL,
                parent_id TEXT,
                status TEXT DEFAULT 'completed'
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_traces_conversation
            ON execution_traces(conversation_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_traces_type
            ON execution_traces(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_traces_timestamp
            ON execution_traces(timestamp)
        """)
        conn.commit()
        conn.close()

    def start_span(
        self,
        conversation_id: str,
        event_type: EventType,
        data: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> str:
        """
        Start a new trace span.

        Args:
            conversation_id: The conversation this trace belongs to
            event_type: Type of event
            data: Event data
            parent_id: Optional parent span for hierarchical traces

        Returns:
            The span ID
        """
        span_id = str(uuid.uuid4())[:8]

        trace = ExecutionTrace(
            id=span_id,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            event_type=event_type,
            data=data,
            parent_id=parent_id,
            status="pending",
        )

        with self._lock:
            self._active_spans[span_id] = trace
            self._conversation_traces[conversation_id].append(trace)

        return span_id

    def end_span(
        self,
        span_id: str,
        result_data: Optional[Dict[str, Any]] = None,
        status: str = "completed"
    ):
        """
        End a trace span and calculate duration.

        Args:
            span_id: The span ID to end
            result_data: Optional result data to merge
            status: Final status (completed, error)
        """
        with self._lock:
            if span_id not in self._active_spans:
                return

            trace = self._active_spans[span_id]
            trace.duration_ms = (datetime.now() - trace.timestamp).total_seconds() * 1000
            trace.status = status

            if result_data:
                trace.data.update(result_data)

            # Save to database
            self._save_trace(trace)

            # Remove from active spans
            del self._active_spans[span_id]

    def _save_trace(self, trace: ExecutionTrace):
        """Save a trace to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            event_type = trace.event_type.value if isinstance(trace.event_type, EventType) else trace.event_type
            cursor.execute("""
                INSERT OR REPLACE INTO execution_traces
                (id, conversation_id, timestamp, event_type, data, duration_ms, parent_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.id,
                trace.conversation_id,
                trace.timestamp.isoformat(),
                event_type,
                json.dumps(trace.data),
                trace.duration_ms,
                trace.parent_id,
                trace.status,
            ))
            conn.commit()
        except Exception as e:
            print(f"Error saving trace: {e}")
        finally:
            conn.close()

    @contextmanager
    def trace(
        self,
        conversation_id: str,
        event_type: EventType,
        data: Dict[str, Any],
        parent_id: Optional[str] = None
    ):
        """
        Context manager for tracing an operation.

        Usage:
            with tracer.trace(conv_id, EventType.TOOL_CALL, {"tool": "read_file"}) as span_id:
                result = execute_tool()
        """
        span_id = self.start_span(conversation_id, event_type, data, parent_id)
        try:
            yield span_id
            self.end_span(span_id, status="completed")
        except Exception as e:
            self.end_span(span_id, {"error": str(e)}, status="error")
            raise

    def get_traces(
        self,
        conversation_id: str,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get traces for a conversation.

        Args:
            conversation_id: The conversation ID
            event_type: Optional filter by event type
            limit: Maximum traces to return

        Returns:
            List of trace dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            if event_type:
                event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
                cursor.execute("""
                    SELECT id, conversation_id, timestamp, event_type, data, duration_ms, parent_id, status
                    FROM execution_traces
                    WHERE conversation_id = ? AND event_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (conversation_id, event_type_str, limit))
            else:
                cursor.execute("""
                    SELECT id, conversation_id, timestamp, event_type, data, duration_ms, parent_id, status
                    FROM execution_traces
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (conversation_id, limit))

            traces = []
            for row in cursor.fetchall():
                traces.append({
                    "id": row[0],
                    "conversation_id": row[1],
                    "timestamp": row[2],
                    "event_type": row[3],
                    "data": json.loads(row[4]) if row[4] else {},
                    "duration_ms": row[5],
                    "parent_id": row[6],
                    "status": row[7],
                })
            return traces

        except Exception as e:
            print(f"Error getting traces: {e}")
            return []
        finally:
            conn.close()

    def get_trace_tree(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get traces organized as a tree structure.

        Returns traces with children nested under their parents.
        """
        traces = self.get_traces(conversation_id, limit=500)

        # Build tree structure
        trace_map = {t["id"]: {**t, "children": []} for t in traces}
        roots = []

        for trace in traces:
            if trace["parent_id"] and trace["parent_id"] in trace_map:
                trace_map[trace["parent_id"]]["children"].append(trace_map[trace["id"]])
            else:
                roots.append(trace_map[trace["id"]])

        return roots

    def get_recent_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent error traces across all conversations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, conversation_id, timestamp, event_type, data, duration_ms
                FROM execution_traces
                WHERE status = 'error' OR event_type = 'error'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            errors = []
            for row in cursor.fetchall():
                errors.append({
                    "id": row[0],
                    "conversation_id": row[1],
                    "timestamp": row[2],
                    "event_type": row[3],
                    "data": json.loads(row[4]) if row[4] else {},
                    "duration_ms": row[5],
                })
            return errors

        except Exception as e:
            print(f"Error getting errors: {e}")
            return []
        finally:
            conn.close()

    def clear_old_traces(self, days: int = 30):
        """Clear traces older than specified days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM execution_traces WHERE timestamp < ?",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print(f"Error clearing traces: {e}")
            return 0
        finally:
            conn.close()


class MetricsCollector:
    """
    Collects and aggregates performance metrics.

    Tracks metrics including:
    - Response latencies
    - Token usage
    - Tool call frequencies
    - Error rates
    - Throughput
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._buffer: List[MetricPoint] = []
        self._buffer_size = 100
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Ensure the metrics table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                tags TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)
        """)
        conn.commit()
        conn.close()

    def record(
        self,
        metric_type: MetricType,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric data point.

        Args:
            metric_type: Type of metric
            metric_name: Specific metric name
            value: Metric value
            tags: Optional tags for filtering
        """
        point = MetricPoint(
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            timestamp=datetime.now(),
        )

        with self._lock:
            self._buffer.append(point)
            if len(self._buffer) >= self._buffer_size:
                self._flush()

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        agent_name: Optional[str] = None
    ):
        """Record a tool call metric."""
        tags = {"tool": tool_name, "success": str(success)}
        if agent_name:
            tags["agent"] = agent_name

        self.record(MetricType.LATENCY, f"tool.{tool_name}.latency", duration_ms, tags)
        self.record(MetricType.TOOL_USAGE, f"tool.{tool_name}.calls", 1, tags)

        if not success:
            self.record(MetricType.ERROR_RATE, f"tool.{tool_name}.errors", 1, tags)

    def record_llm_request(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        agent_name: Optional[str] = None
    ):
        """Record an LLM request metric."""
        tags = {"model": model}
        if agent_name:
            tags["agent"] = agent_name

        self.record(MetricType.LATENCY, "llm.latency", duration_ms, tags)
        self.record(MetricType.TOKEN_COUNT, "llm.tokens_in", tokens_in, tags)
        self.record(MetricType.TOKEN_COUNT, "llm.tokens_out", tokens_out, tags)
        self.record(MetricType.THROUGHPUT, "llm.requests", 1, tags)

    def record_error(
        self,
        error_type: str,
        context: str,
        agent_name: Optional[str] = None
    ):
        """Record an error metric."""
        tags = {"error_type": error_type, "context": context}
        if agent_name:
            tags["agent"] = agent_name

        self.record(MetricType.ERROR_RATE, f"error.{error_type}", 1, tags)

    def _flush(self):
        """Flush buffered metrics to database."""
        if not self._buffer:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            for point in self._buffer:
                metric_type = point.metric_type.value if isinstance(point.metric_type, MetricType) else point.metric_type
                cursor.execute("""
                    INSERT INTO metrics (metric_type, metric_name, value, tags, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric_type,
                    point.metric_name,
                    point.value,
                    json.dumps(point.tags),
                    point.timestamp.isoformat(),
                ))
            conn.commit()
            self._buffer.clear()
        except Exception as e:
            print(f"Error flushing metrics: {e}")
        finally:
            conn.close()

    def flush(self):
        """Public method to flush metrics."""
        with self._lock:
            self._flush()

    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get aggregated metrics for the dashboard.

        Args:
            hours: Time range in hours

        Returns:
            Dashboard data dictionary
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        dashboard = {
            "time_range_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "tool_usage": {},
            "latency_percentiles": {},
            "error_breakdown": {},
            "hourly_activity": [],
        }

        try:
            # Summary stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN metric_name = 'llm.latency' THEN value ELSE 0 END) as total_latency,
                    AVG(CASE WHEN metric_name = 'llm.latency' THEN value ELSE NULL END) as avg_latency,
                    SUM(CASE WHEN metric_name = 'llm.tokens_in' THEN value ELSE 0 END) as total_tokens_in,
                    SUM(CASE WHEN metric_name = 'llm.tokens_out' THEN value ELSE 0 END) as total_tokens_out
                FROM metrics
                WHERE timestamp > ?
            """, (cutoff,))
            row = cursor.fetchone()
            dashboard["summary"] = {
                "total_requests": row[0] or 0,
                "total_latency_ms": row[1] or 0,
                "avg_latency_ms": row[2] or 0,
                "total_tokens_in": row[3] or 0,
                "total_tokens_out": row[4] or 0,
            }

            # Tool usage breakdown
            cursor.execute("""
                SELECT metric_name, SUM(value) as count
                FROM metrics
                WHERE metric_type = 'tool_usage' AND timestamp > ?
                GROUP BY metric_name
                ORDER BY count DESC
                LIMIT 20
            """, (cutoff,))
            for row in cursor.fetchall():
                tool_name = row[0].replace("tool.", "").replace(".calls", "")
                dashboard["tool_usage"][tool_name] = row[1]

            # Latency percentiles
            cursor.execute("""
                SELECT value
                FROM metrics
                WHERE metric_name = 'llm.latency' AND timestamp > ?
                ORDER BY value
            """, (cutoff,))
            latencies = [row[0] for row in cursor.fetchall()]
            if latencies:
                n = len(latencies)
                dashboard["latency_percentiles"] = {
                    "p50": latencies[int(n * 0.5)] if n > 0 else 0,
                    "p90": latencies[int(n * 0.9)] if n > 0 else 0,
                    "p95": latencies[int(n * 0.95)] if n > 0 else 0,
                    "p99": latencies[int(n * 0.99)] if n > 0 else 0,
                    "min": latencies[0] if n > 0 else 0,
                    "max": latencies[-1] if n > 0 else 0,
                }

            # Error breakdown
            cursor.execute("""
                SELECT tags, SUM(value) as count
                FROM metrics
                WHERE metric_type = 'error_rate' AND timestamp > ?
                GROUP BY tags
            """, (cutoff,))
            for row in cursor.fetchall():
                try:
                    tags = json.loads(row[0]) if row[0] else {}
                    error_type = tags.get("error_type", "unknown")
                    dashboard["error_breakdown"][error_type] = row[1]
                except:
                    pass

            # Hourly activity
            cursor.execute("""
                SELECT
                    strftime('%Y-%m-%d %H:00', timestamp) as hour,
                    COUNT(*) as requests,
                    AVG(CASE WHEN metric_name = 'llm.latency' THEN value ELSE NULL END) as avg_latency
                FROM metrics
                WHERE timestamp > ?
                GROUP BY hour
                ORDER BY hour
            """, (cutoff,))
            for row in cursor.fetchall():
                dashboard["hourly_activity"].append({
                    "hour": row[0],
                    "requests": row[1],
                    "avg_latency_ms": row[2] or 0,
                })

            return dashboard

        except Exception as e:
            print(f"Error getting dashboard data: {e}")
            return dashboard
        finally:
            conn.close()

    def get_agent_comparison(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get metrics comparison by agent.

        Returns metrics grouped by agent for comparison.
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT tags, metric_name, SUM(value) as total, AVG(value) as avg
                FROM metrics
                WHERE timestamp > ?
                GROUP BY tags, metric_name
            """, (cutoff,))

            agent_metrics = defaultdict(lambda: {
                "requests": 0,
                "total_latency": 0,
                "avg_latency": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "errors": 0,
            })

            for row in cursor.fetchall():
                try:
                    tags = json.loads(row[0]) if row[0] else {}
                    agent = tags.get("agent", "default")
                    metric_name = row[1]
                    total = row[2]
                    avg = row[3]

                    if "latency" in metric_name:
                        agent_metrics[agent]["total_latency"] += total
                        agent_metrics[agent]["avg_latency"] = avg
                    elif "tokens_in" in metric_name:
                        agent_metrics[agent]["tokens_in"] += total
                    elif "tokens_out" in metric_name:
                        agent_metrics[agent]["tokens_out"] += total
                    elif "requests" in metric_name:
                        agent_metrics[agent]["requests"] += total
                    elif "error" in metric_name:
                        agent_metrics[agent]["errors"] += total
                except:
                    pass

            return [
                {"agent": agent, **metrics}
                for agent, metrics in agent_metrics.items()
            ]

        except Exception as e:
            print(f"Error getting agent comparison: {e}")
            return []
        finally:
            conn.close()

    def clear_old_metrics(self, days: int = 30):
        """Clear metrics older than specified days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print(f"Error clearing metrics: {e}")
            return 0
        finally:
            conn.close()


class ObservabilityEngine:
    """
    Unified observability engine combining tracing and metrics.

    Provides a single interface for all observability features.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tracer = ExecutionTracer(db_path)
        self.metrics = MetricsCollector(db_path)

    @contextmanager
    def trace_operation(
        self,
        conversation_id: str,
        operation_type: str,
        details: Dict[str, Any]
    ):
        """
        Trace an operation with automatic metrics collection.

        Args:
            conversation_id: The conversation ID
            operation_type: Type of operation (tool_call, llm_request, etc.)
            details: Operation details
        """
        event_type = EventType.TOOL_CALL if "tool" in operation_type else EventType.LLM_REQUEST
        start_time = time.time()

        span_id = self.tracer.start_span(conversation_id, event_type, details)

        try:
            yield span_id
            duration_ms = (time.time() - start_time) * 1000
            self.tracer.end_span(span_id, {"duration_ms": duration_ms}, "completed")

            # Record metrics
            if "tool" in operation_type:
                self.metrics.record_tool_call(
                    details.get("tool_name", "unknown"),
                    duration_ms,
                    True,
                    details.get("agent_name"),
                )
            else:
                self.metrics.record_llm_request(
                    details.get("model", "unknown"),
                    details.get("tokens_in", 0),
                    details.get("tokens_out", 0),
                    duration_ms,
                    details.get("agent_name"),
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.tracer.end_span(span_id, {"error": str(e), "duration_ms": duration_ms}, "error")
            self.metrics.record_error(type(e).__name__, operation_type, details.get("agent_name"))
            raise

    def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """Get insights for a specific conversation."""
        traces = self.tracer.get_traces(conversation_id)

        insights = {
            "conversation_id": conversation_id,
            "total_traces": len(traces),
            "total_duration_ms": sum(t.get("duration_ms", 0) or 0 for t in traces),
            "tool_calls": [],
            "errors": [],
            "event_breakdown": defaultdict(int),
        }

        for trace in traces:
            insights["event_breakdown"][trace.get("event_type", "unknown")] += 1

            if trace.get("event_type") == "tool_call":
                insights["tool_calls"].append({
                    "tool": trace.get("data", {}).get("tool_name", "unknown"),
                    "duration_ms": trace.get("duration_ms", 0),
                    "status": trace.get("status", "unknown"),
                })

            if trace.get("status") == "error":
                insights["errors"].append({
                    "event_type": trace.get("event_type"),
                    "error": trace.get("data", {}).get("error", "Unknown error"),
                    "timestamp": trace.get("timestamp"),
                })

        return insights

    def flush(self):
        """Flush all buffered data."""
        self.metrics.flush()

    def cleanup(self, days: int = 30):
        """Clean up old data."""
        traces_deleted = self.tracer.clear_old_traces(days)
        metrics_deleted = self.metrics.clear_old_metrics(days)
        return {"traces_deleted": traces_deleted, "metrics_deleted": metrics_deleted}
