"""Skill loader for parsing and loading agent skills from SKILL.md files.

This module implements Anthropic's agent skills pattern with YAML frontmatter parsing.
Each skill is a directory containing a SKILL.md file with:
- YAML frontmatter (name, description required)
- Markdown instructions for the agent
- Optional supporting files (scripts, configs, etc.)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

# Maximum size for SKILL.md files (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024


class SkillMetadata(TypedDict):
    """Metadata for a skill."""

    name: str
    """Name of the skill."""

    description: str
    """Description of what the skill does."""

    path: str
    """Path to the SKILL.md file."""

    source: str
    """Source of the skill ('user' or 'project')."""


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if a path is safely contained within base_dir.

    This prevents directory traversal attacks via symlinks or path manipulation.
    The function resolves both paths to their canonical form (following symlinks)
    and verifies that the target path is within the base directory.
    """
    try:
        # Resolve both paths to their canonical form (follows symlinks)
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()

        # Check if the resolved path is within the base directory
        # This catches symlinks that point outside the base directory
        resolved_path.relative_to(resolved_base)
        return True  # noqa: TRY300
    except ValueError:
        # Path is not relative to base_dir (outside the directory)
        return False
    except (OSError, RuntimeError):
        # Error resolving paths (e.g., circular symlinks, too many levels)
        return False


def _parse_skill_metadata_from_content(content: str, path_str: str, source: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from raw SKILL.md content."""

    # If content comes from a backend `read` call, it may be formatted with
    # cat-style line numbers (e.g., "     1\t---"). Strip those prefixes
    # before attempting to parse. Plain content is left intact.
    def _strip_cat_numbering(text: str) -> str:
        return "\n".join([re.sub(r"^\s*\d+(?:\.\d+)?\t", "", line) for line in text.splitlines()])

    normalized_content = content

    # Try parsing as-is first.
    # Match YAML frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", normalized_content, re.DOTALL)

    # If no match, try again after stripping cat-style numbering prefixes.
    if not match:
        normalized_content = _strip_cat_numbering(content)
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", normalized_content, re.DOTALL)
    if not match:
        return None

    frontmatter = match.group(1)

    # Parse key-value pairs from YAML (simple parsing, no nested structures)
    metadata: dict[str, str] = {}
    for line in frontmatter.split("\n"):
        # Match "key: value" pattern
        kv_match = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if kv_match:
            key, value = kv_match.groups()
            metadata[key] = value.strip()

    # Validate required fields
    if "name" not in metadata or "description" not in metadata:
        return None

    return SkillMetadata(
        name=metadata["name"],
        description=metadata["description"],
        path=path_str,
        source=source,
    )


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from a SKILL.md file (filesystem)."""
    try:
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            return None
        content = skill_md_path.read_text(encoding="utf-8")
        return _parse_skill_metadata_from_content(content, str(skill_md_path), source)
    except (OSError, UnicodeDecodeError):
        return None


def _list_skills_fs(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """List all skills from a single skills directory using filesystem access."""
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        # Can't resolve base directory, fail safe
        return []

    skills: list[SkillMetadata] = []

    # Iterate through subdirectories
    for skill_dir in skills_dir.iterdir():
        # Security: Catch symlinks pointing outside the skills directory
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        # Look for SKILL.md file
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        # Security: Validate SKILL.md path is safe before reading
        # This catches SKILL.md files that are symlinks pointing outside
        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        # Parse metadata
        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)
    return skills


def _list_skills_backend(skills_dir: str | Path, source: str, backend: BackendProtocol) -> list[SkillMetadata]:
    """List skills using the provided backend (supports virtual mounts)."""
    skills_path = str(Path(skills_dir).expanduser())
    try:
        entries = backend.ls_info(skills_path)
    except Exception:  # noqa: BLE001
        return []

    skills: list[SkillMetadata] = []
    for entry in entries:
        if not entry.get("is_dir", False):
            continue
        skill_dir = entry["path"]
        skill_md_path = f"{skill_dir.rstrip('/')}/SKILL.md"
        try:
            content = backend.read(skill_md_path, offset=0, limit=10000)
        except Exception:  # noqa: BLE001, S112
            continue
        metadata = _parse_skill_metadata_from_content(content, skill_md_path, source=source)
        if metadata:
            skills.append(metadata)
    return skills


def list_skills(
    *,
    user_skills_dir: Path | str | None = None,
    project_skills_dir: Path | str | None = None,
    backend: BackendProtocol | None = None,
) -> list[SkillMetadata]:
    """List skills from user and/or project directories.

    When both directories are provided, project skills with the same name as
    user skills will override them.
    """
    all_skills: dict[str, SkillMetadata] = {}

    # Load user skills first (foundation)
    if user_skills_dir:
        if backend is not None:
            user_skills = _list_skills_backend(user_skills_dir, source="user", backend=backend)
        else:
            user_skills = _list_skills_fs(Path(user_skills_dir), source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    # Load project skills second (override/augment)
    if project_skills_dir:
        if backend is not None:
            project_skills = _list_skills_backend(project_skills_dir, source="project", backend=backend)
        else:
            project_skills = _list_skills_fs(Path(project_skills_dir), source="project")
        for skill in project_skills:
            # Project skills override user skills with the same name
            all_skills[skill["name"]] = skill

    return list(all_skills.values())
