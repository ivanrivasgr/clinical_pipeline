"""
Layer 2 — Clean & Parse
Transforms raw action_text into structured clinical fields.

Handles:
  - Timestamp parsing
  - Severity extraction (✱ count → 1 / 2 / 3 stars)
  - Event type classification via regex patterns
  - Numeric metric + threshold extraction
  - Event state detection (Generated / Ended)
"""

import re
import pandas as pd
import numpy as np


# ─── Event type patterns (order matters: more specific patterns first) ────────

EVENT_PATTERNS = [
    ('VTach',        r'VTach'),
    ('VFib',         r'VFib'),
    ('Asystole',     r'Asystole'),
    ('NonSustainVT', r'Non.?Sustain\s*VT'),
    ('RunPVCs',      r'Run\s*PVCs?'),
    ('IrregularHR',  r'(End\s+)?Irregular\s*HR'),
    ('Brady',        r'\bBrady\b'),
    ('Apnea',        r'\bApnea\b'),
    ('Desat',        r'\bDesat\b'),
    ('NBPs',         r'\bNBPs\b'),
    ('NBPd',         r'\bNBPd\b'),
    ('SpO2',         r'SpO[₂2]'),
    ('HR',           r'\bHR\b'),
    ('RR',           r'\bRR\b'),
    ('TEMP',         r'\bTEMP\b'),
]

# Human-readable severity labels
SEVERITY_LABELS = {1: '⭐', 2: '⭐⭐', 3: '⭐⭐⭐'}

# Action type color mapping (used by dashboard layers)
ACTION_COLOR = {
    'Yellow Alarm':      '#F5A623',
    'Red Alarm':         '#D0021B',
    'Alert Sound':       '#7B8794',
    'Acknowledge':       '#417505',
    'Pause All Alarms':  '#9013FE',
    'Resume All Alarms': '#0070D2',
}

# Event types considered clinically critical
CRITICAL_EVENTS = {'VTach', 'VFib', 'Asystole', 'Desat'}


def clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Layer 2 transformation.
    Input:  raw DataFrame from layer1_ingest.
    Output: clean DataFrame with all parsed clinical fields.
    """
    df = raw_df.copy()

    # 1. Parse timestamps
    df['timestamp'] = pd.to_datetime(
        df['datetime_str'],
        format='%m/%d/%Y %H:%M:%S',
        errors='coerce'
    )

    # Drop rows with unparseable timestamps
    n_before = len(df)
    df = df.dropna(subset=['timestamp']).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[Layer2] Dropped {n_dropped} rows with invalid timestamps")

    # 2. Sort chronologically (ascending)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 3. Extract severity from ✱ star count
    df['severity'] = df['action_text'].apply(_parse_severity)

    # 4. Classify event type
    df['event_type'] = df['action_text'].apply(_parse_event_type)

    # 5. Detect event state (Generated → alarm fired, Ended → alarm cleared)
    df['event_state'] = df['action_text'].apply(_parse_event_state)

    # 6. Extract metric value, comparison condition, and threshold
    parsed_metrics = df['action_text'].apply(_parse_metric)
    df['metric_value'] = parsed_metrics.apply(lambda x: x[0])
    df['condition']    = parsed_metrics.apply(lambda x: x[1])
    df['threshold']    = parsed_metrics.apply(lambda x: x[2])

    # 7. Clean room/bed identifier
    df['room_id'] = df['bed_label'].astype(str).str.strip()

    # 8. Date and hour partitions for BI aggregation
    df['date']    = df['timestamp'].dt.date
    df['hour']    = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.day_name()

    # 9. Human-readable severity label
    df['severity_label'] = df['severity'].map(SEVERITY_LABELS)

    # 10. Flag clinical alarms (exclude system sounds, acknowledges, pauses)
    df['is_alarm'] = df['action_type'].isin(['Yellow Alarm', 'Red Alarm'])

    return df


# ─── Internal parsers ─────────────────────────────────────────────────────────

def _parse_severity(text: str) -> int:
    """Count ✱ characters and return severity 1, 2, or 3."""
    count = len(re.findall(r'✱', str(text)))
    if count == 0:
        return 1
    return min(count, 3)


def _parse_event_type(text: str) -> str:
    """Match action text against EVENT_PATTERNS. Returns first match or 'Other'."""
    for name, pattern in EVENT_PATTERNS:
        if re.search(pattern, str(text), re.IGNORECASE):
            return name
    return 'Other'


def _parse_event_state(text: str) -> str:
    """
    Detect whether the event was Generated (alarm fired) or Ended (alarm cleared).
    Returns 'Unknown' for rows that don't match either state.
    """
    t = str(text)
    if 'Ended'     in t: return 'Ended'
    if 'Generated' in t: return 'Generated'
    return 'Unknown'


def _parse_metric(text: str) -> tuple:
    """
    Extract numeric metric value, comparison condition (> or <), and threshold.
    Example: '✱✱HR  166 >160' → (166, '>', 160)
    Returns: (value, condition, threshold) — or (None, None, None) if not found.
    """
    m = re.search(r'(\d+)\s*([><])\s*(\d+)', str(text))
    if m:
        try:
            return int(m.group(1)), m.group(2), int(m.group(3))
        except ValueError:
            pass
    return None, None, None


def get_clean_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary of the cleaned dataset for logging or UI display."""
    alarms = df[df['is_alarm']]
    return {
        'total_events':      len(df),
        'total_alarms':      len(alarms),
        'rooms':             df['room_id'].nunique(),
        'date_range':        (str(df['date'].min()), str(df['date'].max())),
        'severity_counts':   alarms['severity'].value_counts().to_dict(),
        'event_type_counts': alarms['event_type'].value_counts().to_dict(),
        'critical_alarms':   len(alarms[alarms['severity'] == 3]),
    }
