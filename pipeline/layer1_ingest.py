"""
Layer 1 — Raw Ingestion
Handles CSV and Excel files exported from PIC iX monitoring systems.
Supports the XML-derived CSV format (columns as /Row/@ColumnName).
"""

import pandas as pd
from pathlib import Path
from typing import Union


# ─── Column normalization map ─────────────────────────────────────────────────

COLUMN_MAP = {
    # XML-derived CSV format (PIC iX export)
    '/Row/@Action':              'action_text',
    '/Row/@Action_x0020_Type':   'action_type',
    '/Row/@Bed_x0020_Label':     'bed_label',
    '/Row/@Clinical_x0020_User': 'clinical_user',
    '/Row/@Date':                'datetime_str',
    '/Row/@Device_x0020_Name':   'device_name',
    '/Row/@Institution':         'institution',
    # Excel column names (as seen in spreadsheet screenshots)
    'Action':        'action_text',
    'Action Type':   'action_type',
    'Bed Label':     'bed_label',
    'Clinical User': 'clinical_user',
    'Date':          'datetime_str',
    'Device Name':   'device_name',
    'Institution':   'institution',
}


def load_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV or Excel file from the PIC iX monitoring export.
    Auto-detects format and normalizes column names.

    Supported extensions: .csv, .xlsx, .xls, .xlm
    Returns a raw DataFrame with standardized column names.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in ('.xlsx', '.xls', '.xlm'):
        df = _load_excel(filepath)
    elif ext == '.csv':
        df = _load_csv(filepath)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported formats: .csv, .xlsx, .xls, .xlm"
        )

    df = _normalize_columns(df)
    df = _validate_required_columns(df)
    return df


def _load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load CSV, auto-detecting whether the first row is the export wrapper.
    PIC iX exports include a '/ExportedAuditDataTable' header row that must be skipped.
    """
    with open(filepath, encoding='utf-8-sig', errors='replace') as f:
        first_line = f.readline().strip()

    skip = 1 if first_line.startswith('/ExportedAuditDataTable') else 0
    return pd.read_csv(filepath, skiprows=skip, encoding='utf-8-sig', low_memory=False)


def _load_excel(filepath: Path) -> pd.DataFrame:
    """Load Excel file. Reads the first sheet by default."""
    return pd.read_excel(filepath, engine='openpyxl')


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using COLUMN_MAP. Unknown columns are kept as-is."""
    rename = {col: COLUMN_MAP[col] for col in df.columns if col in COLUMN_MAP}
    return df.rename(columns=rename)


def _validate_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if any required columns are missing after normalization."""
    required = [
        'action_text', 'action_type', 'bed_label',
        'datetime_str', 'device_name', 'institution'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    return df


def get_file_metadata(df: pd.DataFrame) -> dict:
    """Return basic file metadata for logging or UI display."""
    return {
        'total_rows':   len(df),
        'institutions': df['institution'].dropna().unique().tolist(),
        'beds':         df['bed_label'].dropna().nunique(),
        'date_range':   (
            df['datetime_str'].iloc[-1] if len(df) > 0 else None,
            df['datetime_str'].iloc[0]  if len(df) > 0 else None,
        ),
    }
