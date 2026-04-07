# RealTimeX Upgrade Playbook

Repeatable process to upgrade this fork to the latest upstream `main` while preserving RealTimeX customizations.

## Branch Model

| Branch | Role |
|---|---|
| `main` | Synced upstream baseline |
| `realtimex` | Long-lived RealTimeX branch |
| `realtimex-upgrade-<version>` | Short-lived upgrade branch (created per upgrade) |

## Customization Scope

All RealTimeX changes live under `libs/deepagents/`. After a successful upgrade, **only** these files should differ from `origin/main`:

| File (relative to `libs/deepagents/`) | Customization |
|---|---|
| `deepagents/__init__.py` | Exports `create_realtimex_deep_agent` |
| `deepagents/realtimex_graph.py` | RealTimeX agent factory with `prompt` alias |
| `deepagents/backends/composite.py` | Cross-platform path normalization (`\` → `/`) |
| `deepagents/backends/filesystem.py` | Cross-platform path normalization (`\` → `/`) |
| `pyproject.toml` | Package name: `realtimex-deepagents` |
| `uv.lock` | Package name: `realtimex-deepagents` |

### Invariants

These must hold after every upgrade:

1. `name = "realtimex-deepagents"` in both `pyproject.toml` and `uv.lock`
2. `create_realtimex_deep_agent` exists with `prompt: str | None = None` parameter
3. `replace("\\", "/")` appears in both `composite.py` and `filesystem.py`

---

## Prerequisites

```bash
git fetch origin --prune
git switch realtimex
git status --short --branch   # Must be clean
```

## Procedure

### 1. Create upgrade branch

```bash
git switch -c realtimex-upgrade-<version> realtimex
```

### 2. Merge upstream (no commit yet)

```bash
git merge --no-ff --no-commit main
```

### 3. Establish upstream baseline

Resolve all conflicts to upstream state first:

```bash
git checkout --theirs .
git add -A
git diff --name-only --diff-filter=U   # Expected: empty
```

Optionally hard-reset tracked content to upstream:

```bash
git checkout origin/main -- .
```

Do **not** commit yet.

### 4. Reapply customizations via 3-way review

For each customized file, compare three versions:

```bash
BASE="$(git merge-base realtimex origin/main)"

# Example for one file:
git show "$BASE":libs/deepagents/deepagents/backends/filesystem.py > /tmp/base.py
git show origin/main:libs/deepagents/deepagents/backends/filesystem.py > /tmp/upstream.py
git show realtimex:libs/deepagents/deepagents/backends/filesystem.py > /tmp/realtimex.py
diff -u /tmp/upstream.py /tmp/realtimex.py
```

**Decision per hunk:**

| Hunk type | Action |
|---|---|
| Upstream improvement, no RealTimeX intent | Keep upstream |
| Pure RealTimeX behavior | Reapply onto upstream |
| Overlap (both sides changed) | Manually integrate both |

### Reapply rules by file

**`realtimex_graph.py`** — This file is a copy of upstream `graph.py` with one addition: the `prompt` parameter alias. Upgrade procedure:

1. Copy the new upstream `graph.py` over `realtimex_graph.py`.
2. Rename the function from `create_deep_agent` to `create_realtimex_deep_agent`.
3. Add parameter `prompt: str | None = None` (keyword-only).
4. Apply the `prompt` alias behavioral contract:
   - When `prompt` is provided and `system_prompt` is not: use `prompt` as the **final** system prompt. Do **not** append `BASE_AGENT_PROMPT`.
   - When `system_prompt` is provided: behavior is identical to upstream (base prompt is appended).
   - When neither is provided: behavior is identical to upstream (base prompt only).
5. Update the module docstring and function docstring to reflect the RealTimeX function name and the `prompt` parameter.

**`__init__.py`** — Keep upstream exports. Add `create_realtimex_deep_agent` import and `__all__` entry.

**Backends** (`composite.py`, `filesystem.py`) — Keep upstream logic. Reapply cross-platform path normalization: `replace("\\", "/")` on path strings before comparison/routing.

**Packaging** (`pyproject.toml`, `uv.lock`) — Keep upstream dependencies. Set `name = "realtimex-deepagents"`.

---

## Verification

### Automated

```bash
python3 docs/realtimex/scripts/verify_upgrade.py
```

### Manual checklist

**1. No drift outside deepagents:**
```bash
git diff --name-status origin/main..HEAD -- . ':(exclude)libs/deepagents/**'
```
Expected: empty.

**2. Customization scope:**
```bash
git diff --name-status origin/main..HEAD -- libs/deepagents
```
Expected: only files listed in [Customization Scope](#customization-scope).

**3. Invariant checks:**
```bash
rg -n '^name = "realtimex-deepagents"' libs/deepagents/pyproject.toml libs/deepagents/uv.lock
rg -n "create_realtimex_deep_agent|prompt: str \| None" libs/deepagents/deepagents/realtimex_graph.py
rg -n 'replace\("\\\\", "/"\)' libs/deepagents/deepagents/backends/composite.py libs/deepagents/deepagents/backends/filesystem.py
```

**4. Syntax check:**
```bash
python3 -m py_compile libs/deepagents/deepagents/realtimex_graph.py
```

**5. Project quality gates:**
```bash
make lint
make test
```

---

## Commit and PR

Commit only after all checks pass:

```bash
git add -A
git commit -m "chore(realtimex): upgrade from upstream <version> with preserved customizations"
```

PR review must include:
- Rationale for each remaining diff from `origin/main`
- Evidence that overlapping hunks were integrated (not overwritten)
- Verification output summary

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Too many unexpected diffs | Re-establish baseline: `git checkout origin/main -- .`, then reapply customizations |
| Customized file lost upstream behavior | Re-run 3-way comparison for that file, re-integrate manually |
| Package name mismatch | Fix both `pyproject.toml` and `uv.lock` |

## Completion Criteria

1. Upstream alignment outside `libs/deepagents` is exact
2. `libs/deepagents` diff matches customization scope
3. All invariants hold
4. Quality gates pass
5. Changes are committed with clear rationale
