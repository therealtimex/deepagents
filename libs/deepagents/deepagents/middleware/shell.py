"""Middleware that exposes a basic shell tool to agents."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from deepagents.backends.protocol import BackendFactory, BackendProtocol

logger = logging.getLogger(__name__)

SHELL_TOOL_DESCRIPTION = """Run a shell command on the host machine.

Before executing the command, follow these steps:

1. Path Verification:
   - If you will create files or directories, first use `ls` to verify the parent path exists
   - Use exact paths from `ls` output or user-provided paths; do not guess filenames or paths

2. Command Execution:
   - Always quote file paths that contain spaces (e.g., python "path with spaces/script.py")
   - Prefer absolute paths and avoid `cd` to change directories
   - Execute the command and capture the output

Usage notes:
  - The command parameter is required
  - Commands run in the working directory shown below
  - Output includes stdout/stderr and exit code; large output may be truncated
  - VERY IMPORTANT: You MUST avoid shell search commands like `find` and `grep`. Use the `glob`
    and `grep` tools instead. You MUST avoid shell file-reading commands like `cat`, `head`,
    and `tail`; use `read_file` to read file contents.
  - When issuing multiple commands, use `;` or `&&` instead of newlines
  - Use `&&` when commands depend on each other, `;` when they do not

Examples:
  Good examples:
    - shell(command="python /path/to/script.py")
    - shell(command="pytest /path/to/tests")
    - shell(command="npm install && npm test")

  Bad examples (avoid these):
    - shell(command="cd /path && pytest tests")  # Use absolute paths instead
    - shell(command="cat file.txt")  # Use read_file instead
    - shell(command="find . -name '*.py'")  # Use glob instead
    - shell(command="grep -r 'pattern' .")  # Use grep tool instead
"""

SHELL_SYSTEM_PROMPT = """## Shell Tool `shell`

You have access to a `shell` tool for running commands on the host machine.
Use it for running scripts, tests, and other command-line operations.

- shell: run a shell command (returns combined output and exit code)
- Avoid shell search commands like `find`/`grep`; use `glob` and `grep` tools instead.
- Avoid shell file-reading commands like `cat`/`head`/`tail`; use `read_file` instead.
- Quote paths with spaces and prefer absolute paths (avoid `cd`).
- Use exact paths from `ls` output or user-provided paths; do not guess filenames or paths.
"""


class ShellMiddleware(AgentMiddleware[AgentState, Any]):
    """Provide a simple shell tool for local command execution."""

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory | None = None,
        virtual_path_prefixes: list[str] | None = None,
        working_dir: str | None = None,
        timeout: float = 120.0,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialize ShellMiddleware.

        Args:
            backend: Backend for resolving skill paths. Can be a protocol instance
                or a factory that creates one from the runtime.
            virtual_path_prefixes: List of path prefixes that should be resolved
                through the backend (e.g., ["/skills/"]).
            working_dir: Working directory for shell commands. Defaults to cwd.
            timeout: Maximum time in seconds to wait for command completion.
            max_output_bytes: Maximum number of bytes to capture from output.
            env: Environment variables to pass to the subprocess.
        """
        super().__init__()
        self._backend_or_factory = backend
        self._virtual_path_prefixes = tuple(virtual_path_prefixes or [])
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._tool_name = "shell"
        self._env = env if env is not None else os.environ.copy()
        self._working_dir = working_dir or os.getcwd()  # noqa: PTH109
        self._temp_dir: str | None = None
        self._materialized_roots: dict[str, str] = {}
        self._resolution_failures: dict[str, int] = {}
        self._debug_enabled = True

        description = SHELL_TOOL_DESCRIPTION + f"\n\nWorking directory: {self._working_dir}"

        @tool(self._tool_name, description=description)
        def shell_tool(
            command: str,
            runtime: ToolRuntime[None, AgentState],
        ) -> ToolMessage | str:
            return self._run_command(command, runtime=runtime)

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

    def _build_system_prompt(self) -> str:
        """Build the system prompt section for the shell tool."""
        prompt = SHELL_SYSTEM_PROMPT + f"\n\nWorking directory: `{self._working_dir}`"
        if self._virtual_path_prefixes:
            prefixes = ", ".join(f"`{prefix}`" for prefix in self._normalize_prefixes())
            prompt += f"\nSkill paths under these roots are available for execution: {prefixes}"
        return prompt

    def _debug(self, message: str) -> None:
        if not self._debug_enabled:
            return
        logger.info("[DEBUG] %s", message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject shell tool guidance into the system prompt."""
        has_shell_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == self._tool_name for tool in request.tools)
        if has_shell_tool:
            prompt = self._build_system_prompt()
            request = request.override(system_prompt=request.system_prompt + "\n\n" + prompt if request.system_prompt else prompt)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject shell tool guidance into the system prompt."""
        has_shell_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == self._tool_name for tool in request.tools)
        if has_shell_tool:
            prompt = self._build_system_prompt()
            request = request.override(system_prompt=request.system_prompt + "\n\n" + prompt if request.system_prompt else prompt)
        return await handler(request)

    def _get_backend(self, runtime: ToolRuntime[None, AgentState]) -> BackendProtocol | None:
        """Resolve backend from factory if needed."""
        if callable(self._backend_or_factory):
            try:
                return self._backend_or_factory(runtime)
            except Exception:  # noqa: BLE001
                return None
        return self._backend_or_factory

    def _get_temp_dir(self) -> str:
        """Get or create temp directory for materialized files."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="realtimex-shell-")
            self._debug(f"Created temp dir: {self._temp_dir}")
        return self._temp_dir

    def _normalize_prefixes(self) -> tuple[str, ...]:
        """Normalize prefixes to always start and end with a slash."""
        normalized: list[str] = []
        for prefix in self._virtual_path_prefixes:
            if not prefix.startswith("/"):
                prefix = f"/{prefix}"  # noqa: PLW2901
            if not prefix.endswith("/"):
                prefix = f"{prefix}/"  # noqa: PLW2901
            normalized.append(prefix)
        return tuple(normalized)

    def _match_skill_root(self, path: str) -> str | None:
        """Match a path to its skill root based on configured prefixes."""
        for prefix in self._normalize_prefixes():
            if not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :].lstrip("/")
            if not remainder:
                return prefix.rstrip("/")
            skill_segment = remainder.split("/", 1)[0]
            return f"{prefix.rstrip('/')}/{skill_segment}"
        return None

    def _local_path_for(self, backend_path: str) -> str:
        """Map an absolute backend path to a local temp path."""
        temp_root = Path(self._get_temp_dir())
        return str(temp_root / backend_path.lstrip("/"))

    def _list_files_recursive(
        self,
        backend: BackendProtocol,
        root_path: str,
    ) -> list[str]:
        """Recursively list all files under a backend directory."""
        files: list[str] = []
        stack: list[str] = [root_path]
        seen: set[str] = set()

        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            self._debug(f"Listing backend path: {current}")

            entries = backend.ls_info(current)
            for entry in entries:
                entry_path = entry.get("path", "")
                if not entry_path:
                    continue
                is_dir = entry.get("is_dir")
                if is_dir is None:
                    is_dir = entry_path.endswith("/")
                if is_dir:
                    stack.append(entry_path.rstrip("/"))
                else:
                    files.append(entry_path)

        return files

    def _materialize_virtual_path(  # noqa: PLR0911
        self,
        virtual_path: str,
        backend: BackendProtocol,
    ) -> tuple[str | None, str | None]:
        """Download a file or directory to the temp directory.

        Args:
            virtual_path: The backend path (e.g., /skills/sum-two/script.py).
            backend: The backend to download from.

        Returns:
            Tuple of (temp_path, error). On success, temp_path is set and error is None.
            On failure, temp_path is None and error contains the error message.
        """
        self._debug(f"Resolve path: {virtual_path}")
        try:
            entries = backend.ls_info(virtual_path)
        except Exception as exc:  # noqa: BLE001
            self._debug(f"ls_info error for {virtual_path}: {exc}")
            entries = []

        is_dir = any(
            entry.get("is_dir") is True or entry.get("path", "").endswith("/")
            for entry in entries
        )
        if is_dir:
            self._debug(f"Detected directory: {virtual_path}")
            return self._materialize_directory(virtual_path, backend)

        try:
            responses = backend.download_files([virtual_path])
        except Exception as e:  # noqa: BLE001
            return None, f"Failed to access {virtual_path}: {e}"

        if not responses:
            return None, f"No response for {virtual_path}"

        response = responses[0]
        if response.error == "is_directory":
            return self._materialize_directory(virtual_path, backend)
        if response.error:
            return None, f"File not found: {virtual_path}"
        if response.content is None:
            return None, f"Empty content for {virtual_path}"

        dest = Path(self._local_path_for(virtual_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        self._debug(f"Materialized file: {virtual_path} -> {dest}")
        return str(dest), None

    def _materialize_directory(
        self,
        directory_path: str,
        backend: BackendProtocol,
    ) -> tuple[str | None, str | None]:
        """Download all files under a directory into the temp directory."""
        file_paths = self._list_files_recursive(backend, directory_path)
        self._debug(f"Materializing directory: {directory_path} ({len(file_paths)} files)")

        # Ensure the directory exists locally even if empty
        local_root = Path(self._local_path_for(directory_path))
        local_root.mkdir(parents=True, exist_ok=True)

        if not file_paths:
            self._debug(f"Empty directory materialized: {directory_path} -> {local_root}")
            return str(local_root), None

        try:
            responses = backend.download_files(file_paths)
        except Exception as e:  # noqa: BLE001
            return None, f"Failed to download files under {directory_path}: {e}"

        for response in responses:
            if response.error:
                return None, f"File not found: {response.path}"
            if response.content is None:
                return None, f"Empty content for {response.path}"
            dest = Path(self._local_path_for(response.path))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
            self._debug(f"Materialized file: {response.path} -> {dest}")

        return str(local_root), None

    def _resolve_command(
        self,
        command: str,
        runtime: ToolRuntime[None, AgentState],
    ) -> tuple[str, str | None]:
        """Resolve virtual paths in command.

        Args:
            command: The original command with potential virtual paths.
            runtime: The tool runtime for backend resolution.

        Returns:
            Tuple of (resolved_command, error). If any virtual path fails to resolve,
            error contains the message and the original command is returned.
        """
        backend = self._get_backend(runtime)
        if not backend or not self._virtual_path_prefixes:
            return command, None

        try:
            parts = shlex.split(command)
        except ValueError:
            return command, None

        for idx, part in enumerate(parts):
            if not part.startswith("/"):
                continue
            root = self._match_skill_root(part)
            if root is None:
                continue
            self._debug(f"Matched skill root: {root} for {part}")
            if root not in self._materialized_roots:
                temp_root, error = self._materialize_virtual_path(root, backend)
                if error:
                    self._debug(f"Materialization error for {root}: {error}")
                    return command, error
                if temp_root:
                    self._materialized_roots[root] = temp_root
                    self._debug(f"Cached materialized root: {root} -> {temp_root}")
            local_root = self._materialized_roots.get(root)
            if local_root:
                relative = part[len(root) :].lstrip("/")
                resolved_path = Path(local_root) / relative if relative else Path(local_root)
                parts[idx] = str(resolved_path)
                self._debug(f"Resolved path: {part} -> {parts[idx]}")

        return shlex.join(parts), None

    def _run_command(
        self,
        command: str,
        *,
        runtime: ToolRuntime[None, AgentState],
    ) -> ToolMessage:
        """Execute a shell command and return the result."""
        if not command or not isinstance(command, str):
            msg = "Shell tool expects a non-empty command string."
            raise ToolException(msg)

        tool_call_id = runtime.tool_call_id

        # Resolve skill paths in command
        resolved_command, resolve_error = self._resolve_command(command, runtime)
        if resolve_error:
            failures = self._resolution_failures.get(command, 0) + 1
            self._resolution_failures[command] = failures
            if failures >= 2:
                return ToolMessage(
                    content=(f"{resolve_error}\n\nStop retrying the same command. List the directory and use the exact filename from the listing."),
                    tool_call_id=tool_call_id,
                    name=self._tool_name,
                    status="error",
                )
            return ToolMessage(
                content=f"{resolve_error}\n\nCheck the file path and try again with the correct path.",
                tool_call_id=tool_call_id,
                name=self._tool_name,
                status="error",
            )

        try:
            self._debug(f"Executing command: {resolved_command}")
            result = subprocess.run(  # noqa: S602
                resolved_command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=self._env,
                cwd=self._working_dir,
            )

            output_parts: list[str] = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines:
                    output_parts.append(f"[stderr] {line}")  # noqa: PERF401

            output = "\n".join(output_parts) if output_parts else "<no output>"

            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."

            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
                status = "error"
            else:
                status = "success"

        except subprocess.TimeoutExpired:
            output = f"Error: Command timed out after {self._timeout:.1f} seconds."
            status = "error"

        return ToolMessage(
            content=output,
            tool_call_id=tool_call_id,
            name=self._tool_name,
            status=status,
        )


__all__ = ["ShellMiddleware"]
