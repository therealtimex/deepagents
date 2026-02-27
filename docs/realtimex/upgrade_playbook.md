# RealTimeX Upgrade Playbook

## Purpose
Define a production-grade, repeatable process to upgrade this fork to the latest upstream `main` while preserving intentional RealTimeX behavior.

## Scope
This playbook covers:
- Branch preparation and upgrade workflow
- Conflict resolution strategy when upstream and RealTimeX modify the same files
- Rules for customization preservation
- Verification required before merge

This playbook does not depend on personal/local folders or ad-hoc scripts.

## Audience
Human engineers and AI agents with no prior repository context.

## Repository Model

### Branches
- `main`: synced upstream baseline
- `realtimex`: long-lived RealTimeX branch
- `realtimex-upgrade-<version>`: short-lived upgrade branch

### Primary customization package
- `libs/deepagents/`

## Canonical RealTimeX Customization Set

After a successful upgrade, only these files should differ from `origin/main`:

- `libs/deepagents/deepagents/__init__.py`
- `libs/deepagents/deepagents/backends/composite.py`
- `libs/deepagents/deepagents/backends/filesystem.py`
- `libs/deepagents/deepagents/middleware/__init__.py`
- `libs/deepagents/deepagents/middleware/shell.py`
- `libs/deepagents/deepagents/realtimex_graph.py`
- `libs/deepagents/pyproject.toml`
- `libs/deepagents/uv.lock`

Required invariants:
- package name is `realtimex-deepagents` in both `pyproject.toml` and `uv.lock`
- `create_realtimex_deep_agent` exists and stays aligned with upstream `graph.py` behavior except intentional RealTimeX deltas
- Shell middleware support is preserved
- cross-platform path normalization patches are preserved in `composite.py` and `filesystem.py`

## Canonical Customization Reference

Use this section as the authoritative long-term reference for what must be preserved and why.

| File | Intended RealTimeX Modification | Rationale |
|---|---|---|
| `libs/deepagents/deepagents/__init__.py` | Export `create_realtimex_deep_agent`. | Provide stable import surface for RealTimeX callers. |
| `libs/deepagents/deepagents/middleware/__init__.py` | Export `ShellMiddleware`. | Keep middleware discoverable through package-level imports. |
| `libs/deepagents/deepagents/middleware/shell.py` | RealTimeX shell middleware implementation and prompt/tool behavior. | Add shell execution workflow required by RealTimeX runtime behavior. |
| `libs/deepagents/deepagents/realtimex_graph.py` | RealTimeX graph wrapper aligned with upstream `graph.py`, plus `prompt` alias for backward compatibility. | Preserve RealTimeX API compatibility while inheriting upstream graph improvements. |
| `libs/deepagents/deepagents/backends/composite.py` | Cross-platform path normalization (`\\` to `/`) where routing/listing compares paths. | Prevent Windows/macOS path separator mismatches. |
| `libs/deepagents/deepagents/backends/filesystem.py` | Cross-platform path normalization and safe virtual path handling parity for separator variants. | Ensure deterministic behavior across OSes and avoid path parsing regressions. |
| `libs/deepagents/pyproject.toml` | Package name set to `realtimex-deepagents`. | Preserve RealTimeX distribution identity and downstream dependency expectations. |
| `libs/deepagents/uv.lock` | Locked package entry name set to `realtimex-deepagents`, consistent with `pyproject.toml`. | Keep lock metadata consistent with package identity and avoid packaging drift. |

## Prerequisites

```bash
git fetch origin --prune
git switch realtimex
git status --short --branch
```

Ensure working tree is clean before starting upgrade work.

## Upgrade Procedure

### 1. Create upgrade branch
```bash
git switch -c realtimex-upgrade-<version> realtimex
```

### 2. Merge upstream into upgrade branch
```bash
git merge --no-ff --no-commit main
```

Do not commit yet.

### 3. Establish upstream baseline
Resolve conflicts and set baseline to upstream state first:
```bash
git checkout --theirs .
git add -A
git diff --name-only --diff-filter=U
```
Expected: no output.

Optional hard reset of tracked content to upstream baseline:
```bash
git checkout origin/main -- .
```

Still do not commit yet.

## Conflict Resolution and Customization Strategy (Critical)

Do not blindly restore full files from `realtimex` for customized paths when upstream also changed those files.

For each customized file, perform a 3-way review:
- `base`: common ancestor of `realtimex` and `origin/main`
- `upstream`: current `origin/main`
- `realtimex`: current `realtimex`

Get base commit:
```bash
BASE="$(git merge-base realtimex origin/main)"
echo "$BASE"
```

Review one file example:
```bash
git show "$BASE":libs/deepagents/deepagents/backends/filesystem.py > /tmp/base.py
git show origin/main:libs/deepagents/deepagents/backends/filesystem.py > /tmp/upstream.py
git show realtimex:libs/deepagents/deepagents/backends/filesystem.py > /tmp/realtimex.py
diff -u /tmp/upstream.py /tmp/realtimex.py
```

Decision rule per hunk:
- If hunk is upstream improvement with no RealTimeX intent: keep upstream.
- If hunk is pure RealTimeX behavior: reapply onto upstream.
- If hunk overlaps: manually integrate both (upstream semantics + RealTimeX requirement).

This prevents discarding valid upstream fixes while preserving required custom logic.

## Reapply Rules by File Type

### A. API/export glue
- `__init__.py`, `middleware/__init__.py`
- keep upstream exports, add RealTimeX exports (`create_realtimex_deep_agent`, `ShellMiddleware`) only.

### B. `realtimex_graph.py`
- Start from upstream `graph.py` behavior.
- Preserve RealTimeX-only extensions:
  - function name `create_realtimex_deep_agent`
  - `prompt` alias
- Keep upstream evolution (model init behavior, prompt loading, middleware ordering updates unless intentionally overridden).

### C. Backends (`composite.py`, `filesystem.py`)
- Keep upstream logic and security fixes.
- Reapply only RealTimeX cross-platform compatibility patches.

### D. Packaging (`pyproject.toml`, `uv.lock`)
- Keep upstream dependency ecosystem unless intentionally changed.
- Enforce package rename consistency: `realtimex-deepagents`.

## Verification Checklist (Required Before Commit)

### 1. Global upstream alignment outside deepagents
```bash
git diff --name-status origin/main..HEAD -- . ':(exclude)libs/deepagents/**'
```
Expected: empty output.

### 2. Customization scope check
```bash
git diff --name-status origin/main..HEAD -- libs/deepagents
```
Expected: only canonical customization set.

### 3. Invariant checks
```bash
rg -n '^name = "realtimex-deepagents"' libs/deepagents/pyproject.toml libs/deepagents/uv.lock
rg -n "create_realtimex_deep_agent|prompt: str \| None" libs/deepagents/deepagents/realtimex_graph.py
rg -n 'replace\("\\\\", "/"\)' libs/deepagents/deepagents/backends/composite.py libs/deepagents/deepagents/backends/filesystem.py
```

### 4. Syntax sanity
```bash
python3 -m py_compile \
  libs/deepagents/deepagents/realtimex_graph.py \
  libs/deepagents/deepagents/middleware/shell.py
```

### 5. Project validation
Run project-standard quality gates (minimum required by your team policy), for example:
```bash
make lint
make test
```

## Commit and Review

Commit only after all checks pass:
```bash
git add -A
git commit -m "chore(realtimex): upgrade from upstream <version> with preserved customizations"
```

PR review must include:
- why each remaining diff from `origin/main` is intentional
- evidence that overlapping hunks were integrated (not overwritten)
- verification output summary

## Troubleshooting

### Too many unexpected diffs
Re-establish baseline:
```bash
git checkout origin/main -- .
```
Then reapply customizations using 3-way review.

### Customized file lost upstream behavior
Re-run 3-way comparison for that file using `BASE`, `origin/main`, and `realtimex`, then re-integrate manually.

### Package name mismatch
Fix both:
- `libs/deepagents/pyproject.toml`
- `libs/deepagents/uv.lock`

## Completion Criteria
Upgrade is complete when all are true:
1. Upstream alignment outside `libs/deepagents` is exact.
2. `libs/deepagents` diff matches canonical RealTimeX customization set.
3. RealTimeX invariants are satisfied.
4. Project validation gates pass.
5. Changes are committed and reviewable with clear rationale.
