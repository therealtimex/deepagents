# RealTimeX Deep Agent Fork

This repository is a fork of upstream [DeepAgents](../../libs/deepagents/) with RealTimeX-specific customizations. All changes are scoped to `libs/deepagents/`.

## What RealTimeX Customizes

### 1. `realtimex_graph.py` — RealTimeX Agent Factory

A copy of upstream's `graph.py` with the function renamed to `create_realtimex_deep_agent` and one added parameter: `prompt`.

When `prompt` is supplied, it is treated as the **final** system prompt — `BASE_AGENT_PROMPT` is **not** appended. When `system_prompt` is used instead, behavior is identical to upstream (base prompt appended). This supports RealTimeX A2A callers that build the full prompt externally.

```python
from deepagents import create_realtimex_deep_agent

# Using `prompt` — treated as final prompt text (no base prompt appended)
agent = create_realtimex_deep_agent(prompt="Your complete system prompt here")

# Using `system_prompt` — base prompt is appended (upstream behavior)
agent = create_realtimex_deep_agent(system_prompt="Custom prefix instructions")
```

All other parameters and middleware stack are identical to upstream `create_deep_agent`.

### 2. Package Identity

| Field | Upstream | RealTimeX |
|---|---|---|
| Package name | `deepagents` | `realtimex-deepagents` |
| Files | `pyproject.toml`, `uv.lock` | Both must use `realtimex-deepagents` |

### 3. Cross-Platform Path Normalization

Patches in `backends/composite.py` and `backends/filesystem.py` normalize backslash (`\`) path separators to forward slashes (`/`). This prevents path mismatches on Windows/macOS.

## Customized File Set

After any upgrade, **only** these files should differ from `origin/main`:

| File | Purpose |
|---|---|
| `deepagents/__init__.py` | Export `create_realtimex_deep_agent` |
| `deepagents/realtimex_graph.py` | RealTimeX agent factory with `prompt` alias |
| `deepagents/backends/composite.py` | Cross-platform path normalization |
| `deepagents/backends/filesystem.py` | Cross-platform path normalization |
| `pyproject.toml` | Package name `realtimex-deepagents`, version, dependencies |
| `uv.lock` | Locked package name `realtimex-deepagents` |

All paths are relative to `libs/deepagents/`.

## Documentation

| Document | Description |
|---|---|
| [Upgrade Playbook](upgrade_playbook.md) | Step-by-step process for syncing with upstream |
| [Verification Script](scripts/verify_upgrade.py) | Automated invariant checks |

## Quick Start: Verify Upgrade Invariants

```bash
# Against current HEAD vs origin/main
python3 docs/realtimex/scripts/verify_upgrade.py

# Against a specific ref
python3 docs/realtimex/scripts/verify_upgrade.py --base-ref origin/main --target-ref realtimex
```
