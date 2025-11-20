"""
utils.py

Small utility helpers shared by multiple pipeline components.
"""

from typing import Optional


def fmt_float(x: Optional[float]) -> str:
    """
    WHAT:
        Format a float for CSV or debug output.

    WHY:
        We want consistent formatting and to easily handle None values.

    Returns:
        - "" (empty string) if x is None
        - a string with 6 decimal places otherwise
    """
    return "" if x is None else f"{x:.6f}"
