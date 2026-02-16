# RealTimeX Docs

This directory contains RealTimeX-specific engineering process documentation and upgrade validation tooling.

## Contents

- `upgrade_playbook.md`: Authoritative upgrade process for syncing with upstream while preserving RealTimeX customizations.
- `scripts/verify_upgrade.py`: Automated verification script for upgrade invariants.

## Quick Start

Run upgrade verification against current branch:

```bash
python3 docs/realtimex/scripts/verify_upgrade.py
```

Run verification against a specific target ref:

```bash
python3 docs/realtimex/scripts/verify_upgrade.py --target-ref realtimex
```
