"""Middleware that exposes a basic shell tool to agents."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException

from deepagents.backends.protocol import BackendFactory, BackendProtocol


class ShellMiddleware(AgentMiddleware[AgentState, Any]):
    """Provide a simple shell tool for local command execution.

    This middleware resolves virtual paths (e.g., /skills/...) to real files
    before executing shell commands.
    """

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
            backend: Backend for resolving virtual paths. Can be a protocol instance
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

        description = (
            "Execute a shell command. Commands run in a fresh shell with "
            f"working directory: {self._working_dir}. "
            "Virtual paths (e.g., /skills/...) are automatically resolved."
        )

        @tool(self._tool_name, description=description)
        def shell_tool(
            command: str,
            runtime: ToolRuntime[None, AgentState],
        ) -> ToolMessage | str:
            return self._run_command(command, runtime=runtime)

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

    def _resolve_backend(self, runtime: ToolRuntime[None, AgentState]) -> BackendProtocol | None:
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
        return self._temp_dir

    def _materialize_virtual_path(
        self,
        virtual_path: str,
        backend: BackendProtocol,
    ) -> tuple[str | None, str | None]:
        """Download virtual file to temp directory.

        Args:
            virtual_path: The virtual path (e.g., /skills/sum-two/script.py).
            backend: The backend to download from.

        Returns:
            Tuple of (temp_path, error). On success, temp_path is set and error is None.
            On failure, temp_path is None and error contains the error message.
        """
        try:
            responses = backend.download_files([virtual_path])
        except Exception as e:  # noqa: BLE001
            return None, f"Failed to download {virtual_path}: {e}"

        if not responses:
            return None, f"No response for {virtual_path}"

        response = responses[0]
        if response.error:
            return None, f"File not found: {virtual_path}"

        if response.content is None:
            return None, f"Empty content for {virtual_path}"

        # Write to temp location
        temp_dir = self._get_temp_dir()
        filename = Path(virtual_path).name
        dest = Path(temp_dir) / filename
        dest.write_bytes(response.content)

        return str(dest), None

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
        backend = self._resolve_backend(runtime)
        if not backend or not self._virtual_path_prefixes:
            return command, None

        try:
            parts = shlex.split(command)
        except ValueError:
            return command, None

        for idx, part in enumerate(parts):
            if not part.startswith("/"):
                continue
            for prefix in self._virtual_path_prefixes:
                if part.startswith(prefix):
                    temp_path, error = self._materialize_virtual_path(part, backend)
                    if error:
                        return command, error
                    if temp_path:
                        parts[idx] = temp_path
                    break

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

        # Resolve virtual paths in command
        resolved_command, resolve_error = self._resolve_command(command, runtime)
        if resolve_error:
            return ToolMessage(
                content=f"{resolve_error}\n\nCheck the file path and try again with the correct path.",
                tool_call_id=tool_call_id,
                name=self._tool_name,
                status="error",
            )

        try:
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
