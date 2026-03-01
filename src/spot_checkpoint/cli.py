"""
CLI for spot-checkpoint management.

Usage:
    spot-checkpoint list --bucket my-bucket --job-id ccsd-h2o
    spot-checkpoint info --bucket my-bucket --job-id ccsd-h2o
    spot-checkpoint gc --bucket my-bucket --job-id ccsd-h2o --keep 3
"""

from __future__ import annotations

# TODO: Implement CLI using typer
# This is a stub — full implementation follows the build order in CLAUDE.md


def app() -> None:
    """Spot-checkpoint CLI entry point."""
    print("spot-checkpoint CLI — not yet implemented")
    print("See CLAUDE.md build order: cli.py is step 6")


if __name__ == "__main__":
    app()
