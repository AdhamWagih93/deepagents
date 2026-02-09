"""
Agentic Skills System for DeepAgents

This module provides a dynamic skills system that allows users to create,
manage, and assign skills to agents. Skills can be prompt templates, tool
bundles, or multi-step workflows, all powered by local Ollama for generation.
"""

import json
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SkillType(str, Enum):
    """Types of skills available."""
    PROMPT_TEMPLATE = "prompt_template"  # System prompt enhancement
    TOOL_BUNDLE = "tool_bundle"          # Pre-configured tool set
    WORKFLOW = "workflow"                # Multi-step procedure


@dataclass
class Skill:
    """Represents an agentic skill."""
    id: str
    name: str
    description: str
    skill_type: SkillType
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    is_builtin: bool = False
    ollama_generated: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type.value if isinstance(self.skill_type, SkillType) else self.skill_type,
            "content": self.content,
            "tags": self.tags,
            "is_builtin": self.is_builtin,
            "ollama_generated": self.ollama_generated,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        skill_type = data.get("skill_type", "prompt_template")
        if isinstance(skill_type, str):
            skill_type = SkillType(skill_type)

        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Unnamed Skill"),
            description=data.get("description", ""),
            skill_type=skill_type,
            content=data.get("content", {}),
            tags=data.get("tags", []),
            is_builtin=data.get("is_builtin", False),
            ollama_generated=data.get("ollama_generated", False),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


# ============================================================================
# Built-in Skills
# ============================================================================

BUILTIN_SKILLS = [
    {
        "id": "code-reviewer",
        "name": "Code Reviewer",
        "description": "Expert code review with security, performance, and best practices analysis",
        "skill_type": "prompt_template",
        "content": {
            "system_prompt_addition": """You are an expert code reviewer. When reviewing code:

1. **Security Analysis**: Check for vulnerabilities (injection, XSS, CSRF, etc.)
2. **Performance**: Identify potential bottlenecks and optimization opportunities
3. **Best Practices**: Ensure code follows language-specific best practices
4. **Readability**: Evaluate naming, structure, and documentation
5. **Error Handling**: Check for proper error handling and edge cases

Format your review with clear sections and severity levels (Critical, High, Medium, Low).
Provide specific line references and concrete suggestions for improvement.""",
            "example_exchanges": [
                {
                    "user": "Review this Python function",
                    "assistant": "I'll analyze this code for security, performance, and best practices...\n\n## Security Analysis\n...\n\n## Performance\n...\n\n## Recommendations\n..."
                }
            ]
        },
        "tags": ["code", "review", "security", "quality"],
        "is_builtin": True,
    },
    {
        "id": "devops-assistant",
        "name": "DevOps Assistant",
        "description": "Infrastructure and DevOps expert with git, kubectl, and container tools",
        "skill_type": "tool_bundle",
        "content": {
            "tools": ["git_command", "kubectl_command", "podman_command", "read_file", "list_directory"],
            "tool_config": {
                "git_command": {"working_dir": "."},
                "kubectl_command": {"read_only": True},
                "podman_command": {"read_only": True},
            },
            "prompt_context": """You are a DevOps expert with access to git, kubectl, and container tools.

When working with infrastructure:
- Always check current state before making changes
- Explain what commands will do before executing
- Use read-only commands for exploration
- Provide clear summaries of findings

Available tools: git for version control, kubectl for Kubernetes, podman for containers."""
        },
        "tags": ["devops", "git", "kubernetes", "containers", "infrastructure"],
        "is_builtin": True,
    },
    {
        "id": "research-agent",
        "name": "Research Agent",
        "description": "Deep research with structured findings and source citations",
        "skill_type": "prompt_template",
        "content": {
            "system_prompt_addition": """You are a thorough research agent. When researching topics:

1. **Gather Information**: Collect relevant data from multiple sources
2. **Analyze**: Look for patterns, contradictions, and key insights
3. **Synthesize**: Combine findings into coherent conclusions
4. **Cite Sources**: Always reference where information came from
5. **Acknowledge Uncertainty**: Clearly state confidence levels

Structure your research with:
- Executive Summary
- Key Findings
- Detailed Analysis
- Sources & References
- Areas for Further Research""",
            "example_exchanges": [
                {
                    "user": "Research the best practices for API design",
                    "assistant": "## Executive Summary\n...\n\n## Key Findings\n1. ...\n\n## Sources\n- ..."
                }
            ]
        },
        "tags": ["research", "analysis", "documentation"],
        "is_builtin": True,
    },
    {
        "id": "file-explorer",
        "name": "File Explorer",
        "description": "Navigate and understand codebases efficiently",
        "skill_type": "tool_bundle",
        "content": {
            "tools": ["read_file", "list_directory", "glob_search", "grep_search"],
            "tool_config": {},
            "prompt_context": """You are a codebase explorer. Your job is to help users navigate and understand code.

When exploring:
- Start with directory structure to understand organization
- Use glob patterns to find relevant files
- Use grep to search for specific patterns
- Read files to understand implementation details
- Provide clear summaries of what you find

Always explain file relationships and code architecture."""
        },
        "tags": ["files", "navigation", "codebase", "exploration"],
        "is_builtin": True,
    },
    {
        "id": "task-planner",
        "name": "Task Planner",
        "description": "Break down complex tasks into manageable steps with tracking",
        "skill_type": "workflow",
        "content": {
            "steps": [
                {
                    "action": "analyze",
                    "prompt": "First, understand the task completely. Identify the goal, constraints, and success criteria."
                },
                {
                    "action": "decompose",
                    "prompt": "Break down the task into smaller, actionable steps. Each step should be clear and achievable."
                },
                {
                    "action": "prioritize",
                    "prompt": "Order the steps by dependencies and priority. Identify critical path items."
                },
                {
                    "action": "execute",
                    "prompt": "Work through each step methodically. Track progress and note any blockers."
                },
                {
                    "action": "verify",
                    "prompt": "Verify each completed step. Ensure the overall goal is achieved."
                }
            ],
            "fallback_behavior": "retry_with_feedback"
        },
        "tags": ["planning", "tasks", "productivity", "workflow"],
        "is_builtin": True,
    },
    {
        "id": "debug-helper",
        "name": "Debug Helper",
        "description": "Systematic debugging with root cause analysis",
        "skill_type": "prompt_template",
        "content": {
            "system_prompt_addition": """You are an expert debugger. When helping debug issues:

1. **Reproduce**: Understand how to reliably reproduce the issue
2. **Isolate**: Narrow down where the problem occurs
3. **Analyze**: Examine the code path and data flow
4. **Hypothesize**: Form theories about the root cause
5. **Test**: Verify hypotheses with targeted investigation
6. **Fix**: Propose minimal, focused fixes
7. **Prevent**: Suggest ways to prevent similar issues

Use systematic elimination to find root causes. Always explain your reasoning.""",
            "example_exchanges": [
                {
                    "user": "My function returns None unexpectedly",
                    "assistant": "Let's debug this systematically:\n\n1. **Reproduction**: ...\n2. **Analysis**: ...\n3. **Root Cause**: ...\n4. **Fix**: ..."
                }
            ]
        },
        "tags": ["debugging", "troubleshooting", "problem-solving"],
        "is_builtin": True,
    },
]


class SkillStore:
    """SQLite-backed skill storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Ensure the skills tables exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                skill_type TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                is_builtin INTEGER DEFAULT 0,
                ollama_generated INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_skills (
                agent_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                PRIMARY KEY (agent_id, skill_id),
                FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_type ON skills(skill_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_builtin ON skills(is_builtin)")

        conn.commit()
        conn.close()

    def save_skill(self, skill: Skill) -> bool:
        """Save or update a skill."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            skill_type = skill.skill_type.value if isinstance(skill.skill_type, SkillType) else skill.skill_type
            cursor.execute("""
                INSERT OR REPLACE INTO skills
                (id, name, description, skill_type, content, tags, is_builtin, ollama_generated, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                skill.id,
                skill.name,
                skill.description,
                skill_type,
                json.dumps(skill.content),
                json.dumps(skill.tags),
                1 if skill.is_builtin else 0,
                1 if skill.ollama_generated else 0,
                skill.created_at.isoformat() if skill.created_at else datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving skill: {e}")
            return False
        finally:
            conn.close()

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, name, description, skill_type, content, tags, is_builtin, ollama_generated, created_at, updated_at
                FROM skills WHERE id = ?
            """, (skill_id,))

            row = cursor.fetchone()
            if row:
                return Skill(
                    id=row[0],
                    name=row[1],
                    description=row[2] or "",
                    skill_type=SkillType(row[3]),
                    content=json.loads(row[4]) if row[4] else {},
                    tags=json.loads(row[5]) if row[5] else [],
                    is_builtin=bool(row[6]),
                    ollama_generated=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    updated_at=datetime.fromisoformat(row[9]) if row[9] else None,
                )
            return None
        except Exception as e:
            print(f"Error getting skill: {e}")
            return None
        finally:
            conn.close()

    def list_skills(
        self,
        skill_type: Optional[SkillType] = None,
        tags: Optional[List[str]] = None,
        include_builtin: bool = True
    ) -> List[Skill]:
        """List skills with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = "SELECT id, name, description, skill_type, content, tags, is_builtin, ollama_generated, created_at, updated_at FROM skills WHERE 1=1"
            params = []

            if skill_type:
                query += " AND skill_type = ?"
                params.append(skill_type.value if isinstance(skill_type, SkillType) else skill_type)

            if not include_builtin:
                query += " AND is_builtin = 0"

            query += " ORDER BY is_builtin DESC, name ASC"

            cursor.execute(query, params)

            skills = []
            for row in cursor.fetchall():
                skill = Skill(
                    id=row[0],
                    name=row[1],
                    description=row[2] or "",
                    skill_type=SkillType(row[3]),
                    content=json.loads(row[4]) if row[4] else {},
                    tags=json.loads(row[5]) if row[5] else [],
                    is_builtin=bool(row[6]),
                    ollama_generated=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    updated_at=datetime.fromisoformat(row[9]) if row[9] else None,
                )

                # Filter by tags if specified
                if tags:
                    if any(tag in skill.tags for tag in tags):
                        skills.append(skill)
                else:
                    skills.append(skill)

            return skills
        except Exception as e:
            print(f"Error listing skills: {e}")
            return []
        finally:
            conn.close()

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill (cannot delete built-in skills)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if it's a built-in skill
            cursor.execute("SELECT is_builtin FROM skills WHERE id = ?", (skill_id,))
            row = cursor.fetchone()
            if row and row[0]:
                print("Cannot delete built-in skills")
                return False

            cursor.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            cursor.execute("DELETE FROM agent_skills WHERE skill_id = ?", (skill_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting skill: {e}")
            return False
        finally:
            conn.close()

    def assign_skill_to_agent(self, agent_id: str, skill_id: str, priority: int = 0) -> bool:
        """Assign a skill to an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO agent_skills (agent_id, skill_id, priority)
                VALUES (?, ?, ?)
            """, (agent_id, skill_id, priority))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error assigning skill: {e}")
            return False
        finally:
            conn.close()

    def remove_skill_from_agent(self, agent_id: str, skill_id: str) -> bool:
        """Remove a skill from an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM agent_skills WHERE agent_id = ? AND skill_id = ?",
                (agent_id, skill_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing skill: {e}")
            return False
        finally:
            conn.close()

    def get_agent_skills(self, agent_id: str) -> List[Skill]:
        """Get all skills assigned to an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT s.id, s.name, s.description, s.skill_type, s.content, s.tags,
                       s.is_builtin, s.ollama_generated, s.created_at, s.updated_at
                FROM skills s
                JOIN agent_skills a ON s.id = a.skill_id
                WHERE a.agent_id = ?
                ORDER BY a.priority DESC
            """, (agent_id,))

            skills = []
            for row in cursor.fetchall():
                skills.append(Skill(
                    id=row[0],
                    name=row[1],
                    description=row[2] or "",
                    skill_type=SkillType(row[3]),
                    content=json.loads(row[4]) if row[4] else {},
                    tags=json.loads(row[5]) if row[5] else [],
                    is_builtin=bool(row[6]),
                    ollama_generated=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    updated_at=datetime.fromisoformat(row[9]) if row[9] else None,
                ))
            return skills
        except Exception as e:
            print(f"Error getting agent skills: {e}")
            return []
        finally:
            conn.close()


class SkillEngine:
    """
    Manages skills library and Ollama-powered generation.

    Provides:
    - CRUD operations for skills
    - Built-in skill library
    - Ollama-powered skill generation
    - Skill application to agent configurations
    """

    def __init__(self, db_path: str, ollama_client=None):
        """
        Initialize the SkillEngine.

        Args:
            db_path: Path to the SQLite database
            ollama_client: Optional OllamaClient instance for AI-powered generation
        """
        self.store = SkillStore(db_path)
        self.ollama_client = ollama_client
        self._load_builtin_skills()

    def _load_builtin_skills(self):
        """Load built-in skills into the database."""
        for skill_data in BUILTIN_SKILLS:
            existing = self.store.get_skill(skill_data["id"])
            if not existing:
                skill = Skill.from_dict(skill_data)
                skill.is_builtin = True
                self.store.save_skill(skill)

    # ========================================================================
    # CRUD Operations
    # ========================================================================

    def create_skill(
        self,
        name: str,
        skill_type: SkillType,
        content: Dict[str, Any],
        description: str = "",
        tags: List[str] = None
    ) -> Optional[Skill]:
        """Create a new skill."""
        skill = Skill(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            skill_type=skill_type,
            content=content,
            tags=tags or [],
            is_builtin=False,
            ollama_generated=False,
        )

        if self.store.save_skill(skill):
            return skill
        return None

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.store.get_skill(skill_id)

    def update_skill(self, skill_id: str, updates: Dict[str, Any]) -> Optional[Skill]:
        """Update an existing skill."""
        skill = self.store.get_skill(skill_id)
        if not skill:
            return None

        if skill.is_builtin:
            print("Cannot modify built-in skills")
            return None

        # Apply updates
        if "name" in updates:
            skill.name = updates["name"]
        if "description" in updates:
            skill.description = updates["description"]
        if "content" in updates:
            skill.content = updates["content"]
        if "tags" in updates:
            skill.tags = updates["tags"]

        skill.updated_at = datetime.now()

        if self.store.save_skill(skill):
            return skill
        return None

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        return self.store.delete_skill(skill_id)

    def list_skills(
        self,
        skill_type: Optional[SkillType] = None,
        tags: Optional[List[str]] = None
    ) -> List[Skill]:
        """List all skills with optional filters."""
        return self.store.list_skills(skill_type, tags)

    def search_skills(self, query: str) -> List[Skill]:
        """Search skills by name, description, or tags."""
        query_lower = query.lower()
        all_skills = self.store.list_skills()

        return [
            skill for skill in all_skills
            if query_lower in skill.name.lower()
            or query_lower in skill.description.lower()
            or any(query_lower in tag.lower() for tag in skill.tags)
        ]

    # ========================================================================
    # Ollama-powered Generation
    # ========================================================================

    def generate_skill_from_description(self, description: str) -> Optional[Skill]:
        """
        Use Ollama to generate a skill from natural language.

        Args:
            description: Natural language description of the desired skill

        Returns:
            Generated Skill, or None if generation fails
        """
        if not self.ollama_client:
            print("Ollama client not available for skill generation")
            return None

        try:
            # Determine skill type from description
            skill_type_prompt = f"""Analyze this skill description and determine the best skill type:

Description: {description}

Skill types:
1. prompt_template - For expertise, personas, or response formatting
2. tool_bundle - For pre-configured sets of tools with specific configs
3. workflow - For multi-step procedures or processes

Respond with ONLY one of: prompt_template, tool_bundle, workflow"""

            messages = [
                {"role": "system", "content": "You are a skill classification expert. Respond with only the skill type."},
                {"role": "user", "content": skill_type_prompt}
            ]

            skill_type_response = ""
            for chunk in self.ollama_client.chat_stream(messages):
                if chunk.get("content"):
                    skill_type_response += chunk["content"]

            skill_type_str = skill_type_response.strip().lower()
            if "tool_bundle" in skill_type_str:
                skill_type = SkillType.TOOL_BUNDLE
            elif "workflow" in skill_type_str:
                skill_type = SkillType.WORKFLOW
            else:
                skill_type = SkillType.PROMPT_TEMPLATE

            # Generate skill content based on type
            content = self._generate_skill_content(description, skill_type)

            # Generate name and tags
            name_prompt = f"""Generate a short, descriptive name (2-4 words) for this skill:

Description: {description}

Respond with ONLY the skill name, nothing else."""

            messages = [
                {"role": "system", "content": "Generate a concise skill name."},
                {"role": "user", "content": name_prompt}
            ]

            name_response = ""
            for chunk in self.ollama_client.chat_stream(messages):
                if chunk.get("content"):
                    name_response += chunk["content"]

            name = name_response.strip()[:50]  # Limit name length

            # Generate tags
            tags_prompt = f"""Generate 3-5 relevant tags for this skill:

Description: {description}
Name: {name}

Respond with comma-separated tags only, no explanation."""

            messages = [
                {"role": "system", "content": "Generate skill tags."},
                {"role": "user", "content": tags_prompt}
            ]

            tags_response = ""
            for chunk in self.ollama_client.chat_stream(messages):
                if chunk.get("content"):
                    tags_response += chunk["content"]

            tags = [tag.strip().lower() for tag in tags_response.split(",")][:5]

            # Create the skill
            skill = Skill(
                id=str(uuid.uuid4())[:8],
                name=name,
                description=description,
                skill_type=skill_type,
                content=content,
                tags=tags,
                is_builtin=False,
                ollama_generated=True,
            )

            if self.store.save_skill(skill):
                return skill
            return None

        except Exception as e:
            print(f"Error generating skill: {e}")
            return None

    def _generate_skill_content(self, description: str, skill_type: SkillType) -> Dict[str, Any]:
        """Generate skill content based on type."""
        if not self.ollama_client:
            return {}

        try:
            if skill_type == SkillType.PROMPT_TEMPLATE:
                prompt = f"""Create a system prompt addition for an AI assistant based on this description:

{description}

The prompt should:
- Define expertise and approach
- Include specific instructions
- Set tone and format expectations

Respond with ONLY the system prompt text, no JSON or formatting."""

                messages = [
                    {"role": "system", "content": "You are an expert prompt engineer."},
                    {"role": "user", "content": prompt}
                ]

                response = ""
                for chunk in self.ollama_client.chat_stream(messages):
                    if chunk.get("content"):
                        response += chunk["content"]

                return {
                    "system_prompt_addition": response.strip(),
                    "example_exchanges": []
                }

            elif skill_type == SkillType.TOOL_BUNDLE:
                # Return a sensible default tool bundle
                return {
                    "tools": ["read_file", "list_directory", "grep_search"],
                    "tool_config": {},
                    "prompt_context": f"You are configured for: {description}"
                }

            elif skill_type == SkillType.WORKFLOW:
                prompt = f"""Create a workflow for this task:

{description}

Define 3-5 steps. Each step should have:
- action: A short action name
- prompt: Instructions for that step

Respond in this exact JSON format:
{{"steps": [{{"action": "step1", "prompt": "..."}}]}}"""

                messages = [
                    {"role": "system", "content": "You are a workflow designer. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ]

                response = ""
                for chunk in self.ollama_client.chat_stream(messages):
                    if chunk.get("content"):
                        response += chunk["content"]

                try:
                    # Try to parse as JSON
                    content = json.loads(response.strip())
                    if "steps" in content:
                        return content
                except json.JSONDecodeError:
                    pass

                # Fallback workflow
                return {
                    "steps": [
                        {"action": "analyze", "prompt": "Understand the task"},
                        {"action": "plan", "prompt": "Create a plan"},
                        {"action": "execute", "prompt": "Execute the plan"},
                        {"action": "verify", "prompt": "Verify results"},
                    ],
                    "fallback_behavior": "retry_with_feedback"
                }

        except Exception as e:
            print(f"Error generating content: {e}")

        return {}

    def suggest_skills_for_task(self, task: str) -> List[Skill]:
        """
        Recommend existing skills for a given task.

        Args:
            task: Description of the task

        Returns:
            List of recommended skills, ordered by relevance
        """
        all_skills = self.store.list_skills()

        if not self.ollama_client:
            # Simple keyword matching fallback
            task_lower = task.lower()
            scored_skills = []

            for skill in all_skills:
                score = 0
                # Check name match
                if any(word in skill.name.lower() for word in task_lower.split()):
                    score += 3
                # Check description match
                if any(word in skill.description.lower() for word in task_lower.split()):
                    score += 2
                # Check tag match
                if any(word in " ".join(skill.tags).lower() for word in task_lower.split()):
                    score += 1

                if score > 0:
                    scored_skills.append((skill, score))

            scored_skills.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scored_skills[:5]]

        # Use Ollama for smarter matching
        try:
            skill_summaries = "\n".join([
                f"- {s.id}: {s.name} - {s.description} (tags: {', '.join(s.tags)})"
                for s in all_skills
            ])

            prompt = f"""Given this task, recommend the most relevant skills:

Task: {task}

Available skills:
{skill_summaries}

List the top 3-5 skill IDs that would help with this task, comma-separated.
Respond with ONLY the skill IDs, nothing else."""

            messages = [
                {"role": "system", "content": "You are a skill recommendation engine."},
                {"role": "user", "content": prompt}
            ]

            response = ""
            for chunk in self.ollama_client.chat_stream(messages):
                if chunk.get("content"):
                    response += chunk["content"]

            skill_ids = [sid.strip() for sid in response.split(",")]
            skill_map = {s.id: s for s in all_skills}

            return [skill_map[sid] for sid in skill_ids if sid in skill_map]

        except Exception as e:
            print(f"Error suggesting skills: {e}")
            return []

    def improve_skill(self, skill_id: str, feedback: str) -> Optional[Skill]:
        """
        Use Ollama to improve a skill based on feedback.

        Args:
            skill_id: ID of the skill to improve
            feedback: User feedback on what to improve

        Returns:
            Improved skill, or None if improvement fails
        """
        skill = self.store.get_skill(skill_id)
        if not skill or skill.is_builtin:
            return None

        if not self.ollama_client:
            return None

        try:
            current_content = json.dumps(skill.content, indent=2)

            prompt = f"""Improve this skill based on the feedback:

Current Skill:
Name: {skill.name}
Type: {skill.skill_type.value}
Content: {current_content}

Feedback: {feedback}

Provide the improved content in the same JSON structure.
Respond with ONLY the JSON content, no explanation."""

            messages = [
                {"role": "system", "content": "You are a skill improvement expert. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]

            response = ""
            for chunk in self.ollama_client.chat_stream(messages):
                if chunk.get("content"):
                    response += chunk["content"]

            try:
                improved_content = json.loads(response.strip())
                skill.content = improved_content
                skill.updated_at = datetime.now()

                if self.store.save_skill(skill):
                    return skill
            except json.JSONDecodeError:
                pass

            return None

        except Exception as e:
            print(f"Error improving skill: {e}")
            return None

    # ========================================================================
    # Skill Application
    # ========================================================================

    def apply_skill(self, skill: Skill, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a skill to an agent configuration.

        Args:
            skill: The skill to apply
            agent_config: Current agent configuration

        Returns:
            Modified agent configuration
        """
        config = agent_config.copy()

        if skill.skill_type == SkillType.PROMPT_TEMPLATE:
            # Append to system prompt
            current_prompt = config.get("system_prompt", "")
            addition = skill.content.get("system_prompt_addition", "")
            if addition:
                config["system_prompt"] = f"{current_prompt}\n\n{addition}".strip()

        elif skill.skill_type == SkillType.TOOL_BUNDLE:
            # Add tools to the configuration
            current_tools = config.get("tools", [])
            new_tools = skill.content.get("tools", [])
            config["tools"] = list(set(current_tools + new_tools))

            # Add tool-specific configs
            tool_config = skill.content.get("tool_config", {})
            config["tool_config"] = {**config.get("tool_config", {}), **tool_config}

            # Add prompt context
            prompt_context = skill.content.get("prompt_context", "")
            if prompt_context:
                current_prompt = config.get("system_prompt", "")
                config["system_prompt"] = f"{current_prompt}\n\n{prompt_context}".strip()

        elif skill.skill_type == SkillType.WORKFLOW:
            # Store workflow configuration for agent execution
            config["workflow"] = skill.content

        return config

    def apply_skills_to_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all assigned skills to an agent configuration."""
        skills = self.store.get_agent_skills(agent_id)

        config = agent_config.copy()
        for skill in skills:
            config = self.apply_skill(skill, config)

        return config

    # ========================================================================
    # Agent Skill Management
    # ========================================================================

    def assign_skill(self, agent_id: str, skill_id: str, priority: int = 0) -> bool:
        """Assign a skill to an agent."""
        return self.store.assign_skill_to_agent(agent_id, skill_id, priority)

    def unassign_skill(self, agent_id: str, skill_id: str) -> bool:
        """Remove a skill from an agent."""
        return self.store.remove_skill_from_agent(agent_id, skill_id)

    def get_agent_skills(self, agent_id: str) -> List[Skill]:
        """Get all skills assigned to an agent."""
        return self.store.get_agent_skills(agent_id)

    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all skills."""
        all_skills = self.store.list_skills()
        tags = set()
        for skill in all_skills:
            tags.update(skill.tags)
        return sorted(list(tags))
