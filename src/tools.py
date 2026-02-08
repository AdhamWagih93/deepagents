"""
LangChain Tools for Agent Operations

This module provides tools for LangChain agents:
- File operations (read, write, list directory)
- Search tools (glob, grep)
- Shell execution
- Todo management
"""

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
import os
import subprocess
import glob as glob_module
import re
import json
from pathlib import Path
from datetime import datetime

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


# ============================================================================
# Tool Input Schemas
# ============================================================================

class ReadFileInput(BaseModel):
    """Input for read_file tool."""
    file_path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class WriteFileInput(BaseModel):
    """Input for write_file tool."""
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")
    encoding: str = Field(default="utf-8", description="File encoding")


class ListDirectoryInput(BaseModel):
    """Input for list_directory tool."""
    path: str = Field(default=".", description="Directory path to list")
    pattern: str = Field(default="*", description="Glob pattern to filter results")


class GlobSearchInput(BaseModel):
    """Input for glob_search tool."""
    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.py')")
    root_dir: str = Field(default=".", description="Root directory to search from")


class GrepSearchInput(BaseModel):
    """Input for grep_search tool."""
    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(default=".", description="File or directory to search in")
    file_pattern: str = Field(default="*", description="Glob pattern to filter files")
    case_sensitive: bool = Field(default=True, description="Case sensitive search")
    context_lines: int = Field(default=2, description="Number of context lines to show")


class ShellExecuteInput(BaseModel):
    """Input for shell_execute tool."""
    command: str = Field(description="Shell command to execute")
    working_dir: str = Field(default=".", description="Working directory")
    timeout: int = Field(default=30, description="Timeout in seconds")


class TodoWriteInput(BaseModel):
    """Input for todo_write tool."""
    todos: List[Dict[str, str]] = Field(
        description="List of todo items with 'content' and 'status' (pending/in_progress/completed)"
    )


class TodoReadInput(BaseModel):
    """Input for todo_read tool."""
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: pending, in_progress, completed, or None for all"
    )


# ============================================================================
# Tool Implementations
# ============================================================================

@tool(args_schema=ReadFileInput)
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            return f"Error: File not found: {file_path}"
        if not path.is_file():
            return f"Error: Not a file: {file_path}"

        with open(path, 'r', encoding=encoding) as f:
            content = f.read()

        # Add line numbers for context
        lines = content.split('\n')
        numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        return f"File: {path}\nLines: {len(lines)}\n\n" + '\n'.join(numbered_lines)

    except Exception as e:
        return f"Error reading file: {e}"


@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)

    Returns:
        Success message or error
    """
    try:
        path = Path(file_path).resolve()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding=encoding) as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to {path}"

    except Exception as e:
        return f"Error writing file: {e}"


@tool(args_schema=ListDirectoryInput)
def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List contents of a directory.

    Args:
        path: Directory path to list
        pattern: Glob pattern to filter results

    Returns:
        Directory listing
    """
    try:
        dir_path = Path(path).resolve()
        if not dir_path.exists():
            return f"Error: Directory not found: {path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        for entry in sorted(dir_path.glob(pattern)):
            if entry.is_dir():
                entries.append(f"  [DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                entries.append(f"  [FILE] {entry.name} ({size:,} bytes)")

        if not entries:
            return f"Directory {dir_path} is empty or no matches for pattern '{pattern}'"

        return f"Directory: {dir_path}\nEntries: {len(entries)}\n\n" + '\n'.join(entries)

    except Exception as e:
        return f"Error listing directory: {e}"


@tool(args_schema=GlobSearchInput)
def glob_search(pattern: str, root_dir: str = ".") -> str:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., '**/*.py' for all Python files)
        root_dir: Root directory to search from

    Returns:
        List of matching file paths
    """
    try:
        root = Path(root_dir).resolve()
        if not root.exists():
            return f"Error: Directory not found: {root_dir}"

        matches = list(root.glob(pattern))
        matches = sorted(matches)[:100]  # Limit results

        if not matches:
            return f"No files found matching pattern '{pattern}' in {root}"

        result = [f"Found {len(matches)} files matching '{pattern}':\n"]
        for match in matches:
            rel_path = match.relative_to(root) if match.is_relative_to(root) else match
            if match.is_dir():
                result.append(f"  [DIR]  {rel_path}/")
            else:
                result.append(f"  [FILE] {rel_path}")

        return '\n'.join(result)

    except Exception as e:
        return f"Error in glob search: {e}"


@tool(args_schema=GrepSearchInput)
def grep_search(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    case_sensitive: bool = True,
    context_lines: int = 2
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: Regex pattern to search for
        path: File or directory to search in
        file_pattern: Glob pattern to filter files
        case_sensitive: Whether search is case sensitive
        context_lines: Number of context lines to show

    Returns:
        Search results with matching lines and context
    """
    try:
        search_path = Path(path).resolve()
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        results = []
        files_searched = 0
        matches_found = 0

        # Get files to search
        if search_path.is_file():
            files = [search_path]
        else:
            files = list(search_path.rglob(file_pattern))
            # Filter out binary files and limit count
            files = [f for f in files if f.is_file() and f.suffix not in ['.pyc', '.exe', '.dll', '.so', '.bin']]
            files = files[:50]  # Limit files

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                files_searched += 1
                file_matches = []

                for i, line in enumerate(lines):
                    if regex.search(line):
                        matches_found += 1
                        # Get context
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)

                        context = []
                        for j in range(start, end):
                            prefix = ">>>" if j == i else "   "
                            context.append(f"{prefix} {j+1:4d}: {lines[j].rstrip()}")

                        file_matches.append('\n'.join(context))

                if file_matches:
                    rel_path = file_path.relative_to(search_path.parent) if search_path.is_dir() else file_path.name
                    results.append(f"\n--- {rel_path} ---\n" + '\n...\n'.join(file_matches[:5]))  # Limit per file

            except Exception:
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        header = f"Search: '{pattern}' in {path}\nFiles searched: {files_searched}, Matches found: {matches_found}\n"
        return header + '\n'.join(results[:20])  # Limit total results

    except Exception as e:
        return f"Error in grep search: {e}"


@tool(args_schema=ShellExecuteInput)
def shell_execute(command: str, working_dir: str = ".", timeout: int = 30) -> str:
    """Execute a shell command.

    Args:
        command: Shell command to execute
        working_dir: Working directory for the command
        timeout: Timeout in seconds

    Returns:
        Command output (stdout and stderr)
    """
    try:
        work_dir = Path(working_dir).resolve()
        if not work_dir.exists():
            return f"Error: Working directory not found: {working_dir}"

        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output_parts = []
        output_parts.append(f"Command: {command}")
        output_parts.append(f"Working Directory: {work_dir}")
        output_parts.append(f"Exit Code: {result.returncode}")

        if result.stdout:
            stdout = result.stdout.strip()
            if len(stdout) > 5000:
                stdout = stdout[:5000] + "\n... [output truncated]"
            output_parts.append(f"\n--- STDOUT ---\n{stdout}")

        if result.stderr:
            stderr = result.stderr.strip()
            if len(stderr) > 2000:
                stderr = stderr[:2000] + "\n... [error truncated]"
            output_parts.append(f"\n--- STDERR ---\n{stderr}")

        return '\n'.join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


# Global todo storage for the session
_todo_storage: List[Dict[str, Any]] = []


@tool(args_schema=TodoWriteInput)
def todo_write(todos: List[Dict[str, str]]) -> str:
    """Write or update the todo list.

    Args:
        todos: List of todo items with 'content' and 'status'

    Returns:
        Confirmation of updated todos
    """
    global _todo_storage

    try:
        # Validate and update todos
        updated_todos = []
        for i, todo in enumerate(todos):
            item = {
                "id": i + 1,
                "content": todo.get("content", ""),
                "status": todo.get("status", "pending"),
                "updated_at": datetime.now().isoformat(),
            }
            if item["status"] not in ["pending", "in_progress", "completed"]:
                item["status"] = "pending"
            updated_todos.append(item)

        _todo_storage = updated_todos

        # Format output
        output = ["Todo list updated:"]
        for todo in _todo_storage:
            status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(todo["status"], "[ ]")
            output.append(f"  {todo['id']}. {status_icon} {todo['content']}")

        return '\n'.join(output)

    except Exception as e:
        return f"Error updating todos: {e}"


@tool(args_schema=TodoReadInput)
def todo_read(status: Optional[str] = None) -> str:
    """Read the current todo list.

    Args:
        status: Filter by status (pending/in_progress/completed) or None for all

    Returns:
        Current todo list
    """
    global _todo_storage

    try:
        todos = _todo_storage

        if status:
            todos = [t for t in todos if t.get("status") == status]

        if not todos:
            if status:
                return f"No todos with status '{status}'"
            return "Todo list is empty"

        output = [f"Todo List ({len(todos)} items):"]
        for todo in todos:
            status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(todo.get("status", "pending"), "[ ]")
            output.append(f"  {todo['id']}. {status_icon} {todo['content']}")

        # Summary
        pending = len([t for t in _todo_storage if t.get("status") == "pending"])
        in_progress = len([t for t in _todo_storage if t.get("status") == "in_progress"])
        completed = len([t for t in _todo_storage if t.get("status") == "completed"])
        output.append(f"\nSummary: {pending} pending, {in_progress} in progress, {completed} completed")

        return '\n'.join(output)

    except Exception as e:
        return f"Error reading todos: {e}"


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """Registry of available tools."""

    # All available tools
    ALL_TOOLS = {
        "read_file": read_file,
        "write_file": write_file,
        "list_directory": list_directory,
        "glob_search": glob_search,
        "grep_search": grep_search,
        "shell_execute": shell_execute,
        "todo_write": todo_write,
        "todo_read": todo_read,
    }

    # Tool categories
    CATEGORIES = {
        "file_operations": ["read_file", "write_file", "list_directory"],
        "search": ["glob_search", "grep_search"],
        "execution": ["shell_execute"],
        "planning": ["todo_write", "todo_read"],
    }

    def __init__(self):
        self._enabled_tools: Dict[str, BaseTool] = {}

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.ALL_TOOLS.get(name)

    def get_tools(self, names: List[str]) -> List[BaseTool]:
        """Get multiple tools by name."""
        tools = []
        for name in names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)
        return tools

    def get_all_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return list(self.ALL_TOOLS.values())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category."""
        tool_names = self.CATEGORIES.get(category, [])
        return self.get_tools(tool_names)

    def get_tool_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all tools."""
        info = {}
        for name, tool in self.ALL_TOOLS.items():
            info[name] = {
                "name": name,
                "description": tool.description if hasattr(tool, 'description') else str(tool),
                "category": next(
                    (cat for cat, tools in self.CATEGORIES.items() if name in tools),
                    "other"
                ),
            }
        return info


# Global registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


# ============================================================================
# Tool Presets
# ============================================================================

TOOL_PRESETS = {
    "minimal": [],
    "read_only": ["read_file", "list_directory", "glob_search", "grep_search"],
    "standard": ["read_file", "write_file", "list_directory", "glob_search", "grep_search", "todo_write", "todo_read"],
    "full": list(ToolRegistry.ALL_TOOLS.keys()),
}
