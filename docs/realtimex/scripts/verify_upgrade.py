#!/usr/bin/env python3
"""Verify RealTimeX upgrade invariants against upstream."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPECTED_DEEPAGENTS_DIFF = {
    "libs/deepagents/deepagents/__init__.py",
    "libs/deepagents/deepagents/backends/composite.py",
    "libs/deepagents/deepagents/backends/filesystem.py",
    "libs/deepagents/deepagents/middleware/__init__.py",
    "libs/deepagents/deepagents/middleware/shell.py",
    "libs/deepagents/deepagents/realtimex_graph.py",
    "libs/deepagents/pyproject.toml",
    "libs/deepagents/uv.lock",
}


def run_git(args: list[str]) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}")
    return result.stdout


def parse_name_status(output: str) -> set[str]:
    """Parse `git diff --name-status` output into resulting path set."""
    paths: set[str] = set()
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            paths.add(parts[2])
        elif len(parts) >= 2:
            paths.add(parts[1])
        else:
            raise RuntimeError(f"Unrecognized name-status line: {raw_line}")
    return paths


def require_contains(path: Path, needle: str, label: str) -> str | None:
    """Validate that a file contains a required string."""
    content = path.read_text(encoding="utf-8")
    if needle not in content:
        return f"{label}: missing `{needle}` in {path}"
    return None


def verify(base_ref: str, target_ref: str) -> list[str]:
    """Run all upgrade invariant checks and return error list."""
    errors: list[str] = []

    diff_output = run_git(["diff", "--name-status", f"{base_ref}..{target_ref}"])
    changed_paths = parse_name_status(diff_output)

    non_deepagents = sorted(path for path in changed_paths if not path.startswith("libs/deepagents/"))
    if non_deepagents:
        errors.append(
            "Non-deepagents drift detected versus upstream:\n"
            + "\n".join(f"  - {path}" for path in non_deepagents)
        )

    deepagents_changed = {path for path in changed_paths if path.startswith("libs/deepagents/")}
    if deepagents_changed != EXPECTED_DEEPAGENTS_DIFF:
        missing = sorted(EXPECTED_DEEPAGENTS_DIFF - deepagents_changed)
        unexpected = sorted(deepagents_changed - EXPECTED_DEEPAGENTS_DIFF)
        msg_lines = ["Deepagents customization scope mismatch:"]
        if missing:
            msg_lines.append("Missing expected paths:")
            msg_lines.extend(f"  - {path}" for path in missing)
        if unexpected:
            msg_lines.append("Unexpected paths:")
            msg_lines.extend(f"  - {path}" for path in unexpected)
        errors.append("\n".join(msg_lines))

    project_name = 'name = "realtimex-deepagents"'
    pyproject = Path("libs/deepagents/pyproject.toml")
    uv_lock = Path("libs/deepagents/uv.lock")
    for path in (pyproject, uv_lock):
        err = require_contains(path, project_name, "Package naming invariant")
        if err:
            errors.append(err)

    realtime_graph = Path("libs/deepagents/deepagents/realtimex_graph.py")
    middleware_init = Path("libs/deepagents/deepagents/middleware/__init__.py")

    realtime_graph_needles = (
        "def create_realtimex_deep_agent(",
        "prompt: str | None = None",
        "enable_shell: bool = True",
        "ShellMiddleware(",
    )
    for needle in realtime_graph_needles:
        err = require_contains(realtime_graph, needle, "RealTimeX graph invariant")
        if err:
            errors.append(err)

    middleware_needles = (
        "from deepagents.middleware.shell import ShellMiddleware",
        '"ShellMiddleware"',
    )
    for needle in middleware_needles:
        err = require_contains(middleware_init, needle, "Shell middleware export invariant")
        if err:
            errors.append(err)

    composite = Path("libs/deepagents/deepagents/backends/composite.py")
    filesystem = Path("libs/deepagents/deepagents/backends/filesystem.py")
    for path in (composite, filesystem):
        err = require_contains(path, 'replace("\\\\", "/")', "Cross-platform path invariant")
        if err:
            errors.append(err)

    return errors


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main", help="Baseline ref to compare against.")
    parser.add_argument("--target-ref", default="HEAD", help="Target ref to validate.")
    args = parser.parse_args()

    try:
        errors = verify(base_ref=args.base_ref, target_ref=args.target_ref)
    except Exception as exc:  # noqa: BLE001
        print(f"Verification failed to run: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("Upgrade verification failed.")
        for idx, error in enumerate(errors, start=1):
            print(f"\n[{idx}] {error}")
        return 1

    print("Upgrade verification passed.")
    print(f"Compared {args.target_ref} against {args.base_ref}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
