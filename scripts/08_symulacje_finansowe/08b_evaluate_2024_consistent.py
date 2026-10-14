"""Compatibility wrapper for current LR-W20-Binomial financial validation."""

from __future__ import annotations

from _financial_suite_runner import run_current_financial_suite


def main() -> None:
    """Run the current financial validation suite."""

    run_current_financial_suite()


if __name__ == "__main__":
    main()
