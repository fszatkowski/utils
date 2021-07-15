from typing import List, Optional

import pandas as pd


def read_md_table(
    md_table_lines: List[str], idx_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Parses markdown table passed as a list of lines read from file into pandas DataFrame.
    If possible, column values are set to numeric.
    :param md_table_lines: List of strings that make up simple markdown table
    :param idx_col: If provided, this column will be set as DataFrame index.

    Example input md_table_lines:
    lines = [
        "|col1|col2|col3|\n",
        "|---|---|---|\n",
        "|1|2|3|\n",
        "|4|5|6|\n"
    ]
    Example output:
        col1  col2  col3
    0   1     2     3
    1   4     5     6

    """

    md_table_lines = [line.strip() for line in md_table_lines if line.strip()]
    rows = [[cell for cell in row.split("|") if cell] for row in md_table_lines]

    header = rows[0]
    table_rows = rows[2:]
    df = pd.DataFrame(table_rows, columns=header)

    for col in df.columns:
        try:
            # Try converting column to integer
            column = df[col].astype(int)
            if df[col] == column.astype(str):
                df[col] = column
            # If it doesn't work try converting to float
            column = df[col].astype(float)
            df[col] = column
        except Exception:
            continue
    if idx_col is not None:
        df = df.set_index(idx_col)
    return df
