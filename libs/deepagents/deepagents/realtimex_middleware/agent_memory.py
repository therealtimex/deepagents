"""Middleware for loading agent-specific long-term memory into the system prompt."""

import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


class AgentMemoryState(AgentState):
    """State for the agent memory middleware."""

    user_memory: NotRequired[str]
    """Personal preferences from agent memory (applies everywhere)."""

    project_memory: NotRequired[str]
    """Project-specific context (loaded from project root)."""


class AgentMemoryStateUpdate(TypedDict):
    """A state update for the agent memory middleware."""

    user_memory: NotRequired[str]
    """Personal preferences from agent memory (applies everywhere)."""

    project_memory: NotRequired[str]
    """Project-specific context (loaded from project root)."""


# Long-term Memory Documentation (identical to CLI)
LONGTERM_MEMORY_SYSTEM_PROMPT = """

## Long-term Memory

Your long-term memory is stored in files on the filesystem and persists across sessions.

**Global Memory Location**: `{agent_dir_absolute}` (displays as `{agent_dir_display}`)
**Workspace Memory Location**: {project_memory_info}

Your system prompt is loaded from TWO sources at startup:
1. **Global agent.md**: `{agent_dir_absolute}/agent.md` - Your personal preferences across all workspaces
2. **Workspace agent.md**: Loaded from the active workspace if available - Workspace-specific instructions

Workspace agent.md is loaded from the configured workspace path when provided.

**When to CHECK/READ memories (CRITICAL - do this FIRST):**
- **At the start of ANY new session**: Check both global and workspace memories
  - Global: `ls {agent_dir_absolute}`
  - Workspace: `ls {project_deepagents_dir}` (if configured)
- **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check workspace memories FIRST, then global
- **When user asks you to do something**: Check if you have workspace-specific guides or examples
- **When user references past work**: Search project memory files for related context

**Memory-first response pattern:**
1. User asks a question → Check workspace directory first: `ls {project_deepagents_dir}`
2. If relevant files exist → Read them with `read_file '{project_deepagents_dir}/[filename]'`
3. Check global memory if needed → `ls {agent_dir_absolute}`
4. Base your answer on saved knowledge supplemented by general knowledge

**When to update memories:**
- **IMMEDIATELY when the user describes your role or how you should behave**
- **IMMEDIATELY when the user gives feedback on your work** - Update memories to capture what was wrong and how to do it better
- When the user explicitly asks you to remember something
- When patterns or preferences emerge (coding styles, conventions, workflows)
- After significant work where context would help in future sessions

**Learning from feedback:**
- When user says something is better/worse, capture WHY and encode it as a pattern
- Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions
- When user says "you should remember X" or "be careful about Y", treat this as HIGH PRIORITY - update memories IMMEDIATELY
- Look for the underlying principle behind corrections, not just the specific mistake

## Deciding Where to Store Memory

When writing or updating agent memory, decide whether each fact, configuration, or behavior belongs in:

### Global Agent File: `{agent_dir_absolute}/agent.md`
→ Describes the agent's **personality, style, and universal behavior** across all workspaces.

**Store here:**
- Your general tone and communication style
- Universal coding preferences (formatting, comment style, etc.)
- General workflows and methodologies you follow
- Tool usage patterns that apply everywhere
- Personal preferences that don't change per-project

**Examples:**
- "Be concise and direct in responses"
- "Always use type hints in Python"
- "Prefer functional programming patterns"

### Workspace Agent File: `{project_deepagents_dir}/agent.md`
→ Describes **how this specific workspace works** and **how the agent should behave here only.**

**Store here:**
- Project-specific architecture and design patterns
- Coding conventions specific to this codebase
- Project structure and organization
- Testing strategies for this project
- Deployment processes and workflows
- Team conventions and guidelines

**Examples:**
- "This workspace uses FastAPI with SQLAlchemy"
- "Tests go in tests/ directory mirroring src/ structure"
- "All API changes require updating OpenAPI spec"

### Workspace Memory Files: `{project_deepagents_dir}/*.md`
→ Use for **workspace-specific reference information** and structured notes.

**Store here:**
- API design documentation
- Architecture decisions and rationale
- Deployment procedures
- Common debugging patterns
- Onboarding information

**Examples:**
- `{project_deepagents_dir}/api-design.md` - REST API patterns used
- `{project_deepagents_dir}/architecture.md` - System architecture overview
- `{project_deepagents_dir}/deployment.md` - How to deploy this workspace

### File Operations:

**Global memory:**
```
ls {agent_dir_absolute}                              # List user memory files
read_file '{agent_dir_absolute}/agent.md'            # Read user preferences
edit_file '{agent_dir_absolute}/agent.md' ...        # Update user preferences
```

**Workspace memory (preferred for workspace-specific information):**
```
ls {project_deepagents_dir}                          # List project memory files
read_file '{project_deepagents_dir}/agent.md'        # Read project instructions
edit_file '{project_deepagents_dir}/agent.md' ...    # Update project instructions
write_file '{project_deepagents_dir}/agent.md' ...  # Create project memory file
```

**Important**:
- Workspace memory files are stored in the configured workspace memory directory
- Always use absolute paths for file operations
- Check workspace memories BEFORE global when answering workspace-specific questions"""


DEFAULT_MEMORY_SNIPPET = """<user_memory>
{user_memory}
</user_memory>

<project_memory>
{project_memory}
</project_memory>"""


class AgentMemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent-specific long-term memory.

    Loads agent memory from configured agent.md files and injects it into the system prompt.
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        *,
        global_agent_path: str | None = None,
        workspace_agent_path: str | None = None,
        system_prompt_template: str | None = None,
    ) -> None:
        # Handle optional global agent path
        if global_agent_path:
            self.user_agent_md = Path(global_agent_path).expanduser()
            self.user_agent_dir = self.user_agent_md.parent
            self.agent_dir_display = str(self.user_agent_dir)
            self.agent_dir_absolute = str(self.user_agent_dir)
        else:
            self.user_agent_md = None
            self.user_agent_dir = None
            self.agent_dir_display = "(not configured)"
            self.agent_dir_absolute = "(not configured)"

        self.project_agent_path = Path(workspace_agent_path).expanduser() if workspace_agent_path else None
        self.project_deepagents_dir = (
            str(self.project_agent_path.parent) if self.project_agent_path else "[project-root]/.deepagents (not in a project)"
        )
        self.project_root_display = (
            str(self.project_agent_path.parent.parent)
            if self.project_agent_path and self.project_agent_path.parent.name == ".deepagents"
            else (str(self.project_agent_path.parent) if self.project_agent_path else None)
        )

        self.system_prompt_template = system_prompt_template or DEFAULT_MEMORY_SNIPPET

    def before_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """Load agent memory from file before agent execution."""
        result: AgentMemoryStateUpdate = {}

        if "user_memory" not in state and self.user_agent_md and self.user_agent_md.exists():
            with contextlib.suppress(OSError, UnicodeDecodeError):
                result["user_memory"] = self.user_agent_md.read_text()

        if "project_memory" not in state and self.project_agent_path and self.project_agent_path.exists():
            with contextlib.suppress(OSError, UnicodeDecodeError):
                result["project_memory"] = self.project_agent_path.read_text()

        return result

    def _build_system_prompt(self, request: ModelRequest) -> str:
        state = cast("AgentMemoryState", request.state)
        user_memory = state.get("user_memory")
        project_memory = state.get("project_memory")
        base_system_prompt = request.system_prompt

        if self.project_agent_path and project_memory:
            project_memory_info = f"`{self.project_root_display}` (detected)" if self.project_root_display else f"`{self.project_agent_path}` (detected)"
        elif self.project_agent_path:
            project_memory_info = f"`{self.project_root_display}` (no agent.md found)" if self.project_root_display else f"`{self.project_agent_path}` (no agent.md found)"
        else:
            project_memory_info = "None (not in a project)"

        memory_section = self.system_prompt_template.format(
            user_memory=user_memory if user_memory else "(No user agent.md)",
            project_memory=project_memory if project_memory else "(No project agent.md)",
        )

        system_prompt = memory_section

        if base_system_prompt:
            system_prompt += "\n\n" + base_system_prompt

        system_prompt += "\n\n" + LONGTERM_MEMORY_SYSTEM_PROMPT.format(
            agent_dir_absolute=self.agent_dir_absolute,
            agent_dir_display=self.agent_dir_display,
            project_memory_info=project_memory_info,
            project_deepagents_dir=self.project_deepagents_dir,
        )

        return system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        system_prompt = self._build_system_prompt(request)
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        system_prompt = self._build_system_prompt(request)
        return await handler(request.override(system_prompt=system_prompt))
