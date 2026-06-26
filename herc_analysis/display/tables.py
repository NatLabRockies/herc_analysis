"""L6 -- great_tables rendering, lifted out of comparison for reuse.

A single reusable ``to_great_table`` so any wide table (a ``Comparison`` table,
a capacity report, etc.) renders the same way.
"""

from __future__ import annotations

import pandas as pd


def to_great_table(
    df: pd.DataFrame,
    *,
    rowname_col: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    decimals: int = 2,
):
    """Render a (possibly MultiIndex-columned) DataFrame as a great_tables table.

    Args:
        df (pd.DataFrame): The table to render. If its columns are a
            ``MultiIndex`` they are flattened to ``"a:b"`` strings.
        rowname_col (str, optional): Column to use as the row-name stub. If None
            the index is reset and used. Defaults to None.
        title (str, optional): Table title. Defaults to None.
        subtitle (str, optional): Table subtitle. Defaults to None.
        decimals (int): Decimal places for numeric columns. Defaults to 2.

    Returns:
        great_tables.GT: The formatted table.

    Raises:
        ImportError: If ``great_tables`` is not installed.
    """
    try:
        from great_tables import GT
    except ImportError as exc:
        raise ImportError(
            "great_tables is required for to_great_table(). Install it with "
            "'pip install great-tables'."
        ) from exc

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [":".join(str(level) for level in col) for col in out.columns]

    if rowname_col is None:
        out = out.reset_index()
        rowname_col = out.columns[0]

    gt = GT(out, rowname_col=rowname_col)
    if title is not None:
        gt = gt.tab_header(title=title, subtitle=subtitle or "")
    num_cols = [c for c in out.columns if c != rowname_col]
    if num_cols:
        gt = gt.fmt_number(columns=num_cols, decimals=decimals)
    return gt
