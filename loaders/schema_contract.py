"""
Canonical schema contract: Simulator → BigQuery → dbt.

Loads contract from dbt seeds/source_schema_contract.csv when available.
VALIDATION ONLY: asserts DataFrame dtypes match contract. No coercion.
If dtype is wrong → crash.
"""

from pathlib import Path

import pandas as pd

# Path to dbt seed (when run from DataModule-DBT repo root)
_SEED_PATH = Path(__file__).resolve().parent.parent.parent / "seeds" / "source_schema_contract.csv"

# table_name -> { column_name: expected_bq_type }
TABLE_CONTRACT: dict[str, dict[str, str]] = {}

if _SEED_PATH.exists():
    _df = pd.read_csv(_SEED_PATH)
    for _, row in _df.iterrows():
        table = row["table_name"]
        col = row["column_name"]
        typ = str(row["expected_bq_type"]).strip().upper()
        if table not in TABLE_CONTRACT:
            TABLE_CONTRACT[table] = {}
        TABLE_CONTRACT[table][col] = typ

# BigQuery type -> allowed pandas dtype names (validation)
_ALLOWED_PANDAS: dict[str, tuple[str, ...]] = {
    "STRING": ("object", "string"),
    "INT64": ("int64", "Int64"),
    "NUMERIC": ("float64",),
    "FLOAT64": ("float64",),
    "BIGNUMERIC": ("float64",),
    "BOOL": ("bool", "boolean"),
    "BOOLEAN": ("bool", "boolean"),
    "TIMESTAMP": ("datetime64[ns]", "datetime64[ns, UTC]"),
    "DATETIME": ("datetime64[ns]", "datetime64[ns, UTC]"),
    "DATE": ("object", "datetime64[ns]", "datetime64[ns, UTC]"),
}


def validate_dataframe_against_contract(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Validate that DataFrame column dtypes match the contract. No modification.
    Raises TypeError with details if any column has wrong dtype.
    Skips validation for empty DataFrames (no rows to check).
    """
    expected = TABLE_CONTRACT.get(table_name, {})
    if not expected or df.empty:
        return df

    errors = []
    for col, exp_bq in expected.items():
        if col not in df.columns:
            continue
        allowed = _ALLOWED_PANDAS.get(exp_bq.upper(), ("object", "string"))
        actual = str(df[col].dtype)
        if actual not in allowed:
            errors.append(f"{table_name}.{col}: dtype {actual} (expected BQ {exp_bq} -> pandas {allowed})")

    if errors:
        raise TypeError("Schema contract violation:\n" + "\n".join(errors))

    return df
