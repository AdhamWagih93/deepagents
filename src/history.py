"""
Interaction History Tracking for DeepAgents

This module provides comprehensive tracking and visualization of agent interactions,
including tool calls, todos, durations, and metrics.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import sqlite3
import os
from pathlib import Path


@dataclass
class ToolCallRecord:
    """Record of a tool call."""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result[:500] if self.result else None,  # Truncate for display
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class TodoRecord:
    """Record of a todo item."""
    content: str
    status: str  # pending, in_progress, completed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    agent_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "agent_name": self.agent_name,
        }


@dataclass
class InteractionRecord:
    """Complete record of an agent interaction."""
    id: str
    session_id: str
    agent_name: str
    agent_type: str
    user_message: str
    assistant_message: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    todos: List[TodoRecord] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    backend_used: str = ""
    middleware_used: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "todos": [t.to_dict() for t in self.todos],
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "backend_used": self.backend_used,
            "middleware_used": self.middleware_used,
            "error": self.error,
            "metadata": self.metadata,
        }


class HistoryStore:
    """SQLite-based store for interaction history."""

    def __init__(self, db_path: str = "deepagents_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_name TEXT,
                agent_type TEXT,
                created_at TEXT,
                config_json TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_name TEXT,
                agent_type TEXT,
                user_message TEXT,
                assistant_message TEXT,
                timestamp TEXT,
                duration_ms REAL,
                tokens_in INTEGER,
                tokens_out INTEGER,
                backend_used TEXT,
                middleware_json TEXT,
                error TEXT,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                interaction_id TEXT,
                name TEXT,
                arguments_json TEXT,
                result TEXT,
                duration_ms REAL,
                timestamp TEXT,
                success INTEGER,
                error TEXT,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT,
                content TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT,
                agent_name TEXT,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            )
        """)

        conn.commit()
        conn.close()

    def save_session(self, session_id: str, agent_name: str, agent_type: str, config: Dict):
        """Save a session record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO sessions (id, agent_name, agent_type, created_at, config_json)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, agent_name, agent_type, datetime.now().isoformat(), json.dumps(config)))

        conn.commit()
        conn.close()

    def save_interaction(self, record: InteractionRecord):
        """Save an interaction record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO interactions (
                id, session_id, agent_name, agent_type, user_message, assistant_message,
                timestamp, duration_ms, tokens_in, tokens_out, backend_used,
                middleware_json, error, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.id, record.session_id, record.agent_name, record.agent_type,
            record.user_message, record.assistant_message, record.timestamp.isoformat(),
            record.duration_ms, record.tokens_in, record.tokens_out, record.backend_used,
            json.dumps(record.middleware_used), record.error, json.dumps(record.metadata)
        ))

        # Save tool calls
        for tc in record.tool_calls:
            cursor.execute("""
                INSERT INTO tool_calls (
                    id, interaction_id, name, arguments_json, result, duration_ms,
                    timestamp, success, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tc.id, record.id, tc.name, json.dumps(tc.arguments), tc.result,
                tc.duration_ms, tc.timestamp.isoformat(), 1 if tc.success else 0, tc.error
            ))

        # Save todos
        for todo in record.todos:
            cursor.execute("""
                INSERT INTO todos (
                    interaction_id, content, status, created_at, completed_at, agent_name
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.id, todo.content, todo.status, todo.created_at.isoformat(),
                todo.completed_at.isoformat() if todo.completed_at else None, todo.agent_name
            ))

        conn.commit()
        conn.close()

    def get_sessions(self, limit: int = 50) -> List[Dict]:
        """Get recent sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, agent_name, agent_type, created_at, config_json
            FROM sessions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "id": row[0],
                "agent_name": row[1],
                "agent_type": row[2],
                "created_at": row[3],
                "config": json.loads(row[4]) if row[4] else {}
            })

        conn.close()
        return sessions

    def get_interactions(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get interactions, optionally filtered by session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute("""
                SELECT id, session_id, agent_name, agent_type, user_message, assistant_message,
                       timestamp, duration_ms, tokens_in, tokens_out, backend_used,
                       middleware_json, error, metadata_json
                FROM interactions
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
                SELECT id, session_id, agent_name, agent_type, user_message, assistant_message,
                       timestamp, duration_ms, tokens_in, tokens_out, backend_used,
                       middleware_json, error, metadata_json
                FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        interactions = []
        for row in cursor.fetchall():
            interaction = {
                "id": row[0],
                "session_id": row[1],
                "agent_name": row[2],
                "agent_type": row[3],
                "user_message": row[4],
                "assistant_message": row[5],
                "timestamp": row[6],
                "duration_ms": row[7],
                "tokens_in": row[8],
                "tokens_out": row[9],
                "backend_used": row[10],
                "middleware_used": json.loads(row[11]) if row[11] else [],
                "error": row[12],
                "metadata": json.loads(row[13]) if row[13] else {},
            }

            # Get tool calls
            cursor.execute("""
                SELECT id, name, arguments_json, result, duration_ms, timestamp, success, error
                FROM tool_calls
                WHERE interaction_id = ?
            """, (row[0],))

            interaction["tool_calls"] = [{
                "id": tc[0],
                "name": tc[1],
                "arguments": json.loads(tc[2]) if tc[2] else {},
                "result": tc[3],
                "duration_ms": tc[4],
                "timestamp": tc[5],
                "success": bool(tc[6]),
                "error": tc[7],
            } for tc in cursor.fetchall()]

            # Get todos
            cursor.execute("""
                SELECT content, status, created_at, completed_at, agent_name
                FROM todos
                WHERE interaction_id = ?
            """, (row[0],))

            interaction["todos"] = [{
                "content": t[0],
                "status": t[1],
                "created_at": t[2],
                "completed_at": t[3],
                "agent_name": t[4],
            } for t in cursor.fetchall()]

            interactions.append(interaction)

        conn.close()
        return interactions

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total counts
        cursor.execute("SELECT COUNT(*) FROM sessions")
        stats["total_sessions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM interactions")
        stats["total_interactions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tool_calls")
        stats["total_tool_calls"] = cursor.fetchone()[0]

        # Error count
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE error IS NOT NULL AND error != ''")
        stats["total_errors"] = cursor.fetchone()[0]

        # Aggregates
        cursor.execute("""
            SELECT SUM(duration_ms), SUM(tokens_in), SUM(tokens_out), AVG(duration_ms),
                   MIN(duration_ms), MAX(duration_ms)
            FROM interactions
        """)
        row = cursor.fetchone()
        stats["total_duration_ms"] = row[0] or 0
        stats["total_tokens_in"] = row[1] or 0
        stats["total_tokens_out"] = row[2] or 0
        stats["avg_duration_ms"] = row[3] or 0
        stats["min_duration_ms"] = row[4] or 0
        stats["max_duration_ms"] = row[5] or 0

        # By agent type
        cursor.execute("""
            SELECT agent_type, COUNT(*) as count, AVG(duration_ms) as avg_duration
            FROM interactions
            GROUP BY agent_type
        """)
        stats["by_agent_type"] = {
            row[0]: {"count": row[1], "avg_duration_ms": row[2]}
            for row in cursor.fetchall()
        }

        # By agent name
        cursor.execute("""
            SELECT agent_name, COUNT(*) as count, AVG(duration_ms) as avg_duration,
                   SUM(tokens_out) as total_tokens
            FROM interactions
            GROUP BY agent_name
            ORDER BY count DESC
        """)
        stats["by_agent_name"] = {
            row[0]: {"count": row[1], "avg_duration_ms": row[2], "total_tokens": row[3] or 0}
            for row in cursor.fetchall()
        }

        # Tool usage
        cursor.execute("""
            SELECT name, COUNT(*) as count, AVG(duration_ms) as avg_duration,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM tool_calls
            GROUP BY name
            ORDER BY count DESC
        """)
        stats["tool_usage"] = {
            row[0]: {
                "count": row[1],
                "avg_duration_ms": row[2],
                "success_rate": row[3] / row[1] * 100 if row[1] > 0 else 0
            }
            for row in cursor.fetchall()
        }

        # Recent activity (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM interactions WHERE timestamp > ?
        """, (yesterday,))
        stats["interactions_last_24h"] = cursor.fetchone()[0]

        # Success rate
        if stats["total_interactions"] > 0:
            stats["success_rate"] = ((stats["total_interactions"] - stats["total_errors"]) /
                                     stats["total_interactions"]) * 100
        else:
            stats["success_rate"] = 100.0

        conn.close()
        return stats

    def get_interactions_timeline(self, days: int = 7) -> List[Dict]:
        """Get interactions grouped by day for timeline chart."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT DATE(timestamp) as date,
                   COUNT(*) as count,
                   AVG(duration_ms) as avg_duration,
                   SUM(tokens_out) as total_tokens,
                   SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) as errors
            FROM interactions
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (start_date,))

        timeline = []
        for row in cursor.fetchall():
            timeline.append({
                "date": row[0],
                "interactions": row[1],
                "avg_duration_ms": row[2] or 0,
                "total_tokens": row[3] or 0,
                "errors": row[4] or 0,
            })

        conn.close()
        return timeline

    def get_hourly_activity(self, days: int = 1) -> List[Dict]:
        """Get interactions grouped by hour for activity chart."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT strftime('%H', timestamp) as hour,
                   COUNT(*) as count
            FROM interactions
            WHERE timestamp > ?
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        """, (start_date,))

        activity = []
        for row in cursor.fetchall():
            activity.append({
                "hour": f"{row[0]}:00",
                "interactions": row[1],
            })

        conn.close()
        return activity

    def get_response_time_distribution(self) -> List[Dict]:
        """Get response time distribution for histogram."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Bucket response times
        cursor.execute("""
            SELECT
                CASE
                    WHEN duration_ms < 1000 THEN '< 1s'
                    WHEN duration_ms < 3000 THEN '1-3s'
                    WHEN duration_ms < 5000 THEN '3-5s'
                    WHEN duration_ms < 10000 THEN '5-10s'
                    WHEN duration_ms < 30000 THEN '10-30s'
                    ELSE '> 30s'
                END as bucket,
                COUNT(*) as count
            FROM interactions
            GROUP BY bucket
            ORDER BY
                CASE bucket
                    WHEN '< 1s' THEN 1
                    WHEN '1-3s' THEN 2
                    WHEN '3-5s' THEN 3
                    WHEN '5-10s' THEN 4
                    WHEN '10-30s' THEN 5
                    ELSE 6
                END
        """)

        distribution = []
        for row in cursor.fetchall():
            distribution.append({
                "bucket": row[0],
                "count": row[1],
            })

        conn.close()
        return distribution

    def search_interactions(self, query: str, limit: int = 50) -> List[Dict]:
        """Search interactions by user message or assistant message."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        search_pattern = f"%{query}%"
        cursor.execute("""
            SELECT id, session_id, agent_name, agent_type, user_message, assistant_message,
                   timestamp, duration_ms, tokens_in, tokens_out, error
            FROM interactions
            WHERE user_message LIKE ? OR assistant_message LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (search_pattern, search_pattern, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "session_id": row[1],
                "agent_name": row[2],
                "agent_type": row[3],
                "user_message": row[4],
                "assistant_message": row[5],
                "timestamp": row[6],
                "duration_ms": row[7],
                "tokens_in": row[8],
                "tokens_out": row[9],
                "error": row[10],
            })

        conn.close()
        return results

    def clear_history(self):
        """Clear all history data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM todos")
        cursor.execute("DELETE FROM tool_calls")
        cursor.execute("DELETE FROM interactions")
        cursor.execute("DELETE FROM sessions")

        conn.commit()
        conn.close()


class InteractionTracker:
    """Real-time tracker for agent interactions."""

    def __init__(self, store: Optional[HistoryStore] = None):
        self.store = store or HistoryStore()
        self.current_session_id: Optional[str] = None
        self.current_interaction: Optional[InteractionRecord] = None
        self._pending_tool_calls: List[ToolCallRecord] = []
        self._pending_todos: List[TodoRecord] = []

    def start_session(self, session_id: str, agent_name: str, agent_type: str, config: Dict):
        """Start a new session."""
        self.current_session_id = session_id
        self.store.save_session(session_id, agent_name, agent_type, config)

    def start_interaction(
        self,
        interaction_id: str,
        agent_name: str,
        agent_type: str,
        user_message: str,
        backend_used: str = "",
        middleware_used: List[str] = None
    ):
        """Start tracking a new interaction."""
        self.current_interaction = InteractionRecord(
            id=interaction_id,
            session_id=self.current_session_id or "",
            agent_name=agent_name,
            agent_type=agent_type,
            user_message=user_message,
            assistant_message="",
            backend_used=backend_used,
            middleware_used=middleware_used or [],
        )
        self._pending_tool_calls = []
        self._pending_todos = []

    def record_tool_call(
        self,
        tool_id: str,
        name: str,
        arguments: Dict[str, Any],
        result: Optional[str] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a tool call."""
        tc = ToolCallRecord(
            id=tool_id,
            name=name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self._pending_tool_calls.append(tc)

    def record_todo(
        self,
        content: str,
        status: str,
        agent_name: str = ""
    ):
        """Record a todo item."""
        todo = TodoRecord(
            content=content,
            status=status,
            agent_name=agent_name,
            completed_at=datetime.now() if status == "completed" else None,
        )
        self._pending_todos.append(todo)

    def complete_interaction(
        self,
        assistant_message: str,
        duration_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Complete and save the current interaction."""
        if not self.current_interaction:
            return

        self.current_interaction.assistant_message = assistant_message
        self.current_interaction.duration_ms = duration_ms
        self.current_interaction.tokens_in = tokens_in
        self.current_interaction.tokens_out = tokens_out
        self.current_interaction.error = error
        self.current_interaction.metadata = metadata or {}
        self.current_interaction.tool_calls = self._pending_tool_calls
        self.current_interaction.todos = self._pending_todos

        self.store.save_interaction(self.current_interaction)
        self.current_interaction = None

    def get_session_history(self) -> List[Dict]:
        """Get history for the current session."""
        if not self.current_session_id:
            return []
        return self.store.get_interactions(self.current_session_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        return self.store.get_statistics()
