"""
Layer 3 — Event-Level Dataset (Core)
Pairs Generated ↔ Ended events to compute duration_sec.
This is the primary analytical table used by all downstream layers.

Output schema:
    room_id, start_time, end_time, duration_sec, severity, event_type,
    metric_value, threshold, condition, alert_text, device_name,
    institution, date, hour, weekday, severity_label, is_noise

Performance:
    Uses pandas merge_asof for event pairing (vectorized C-level matching).
    ~25K events pairs in <5 seconds vs 10+ minutes with row-by-row iteration.
"""

import pandas as pd
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

# Maximum forward-look window (seconds) to find a matching "Ended" for a "Generated"
MAX_PAIRING_WINDOW_SEC = 3600  # 1 hour

# Alarms shorter than this (in seconds) are flagged as noise (no clinical value)
NOISE_THRESHOLD_SEC = 10


def build_event_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for Layer 3.

    Pairing strategy (merge_asof):
    - Filter to clinical alarms only (is_alarm == True)
    - Split into 'Generated' and 'Ended' subsets
    - Sort both by timestamp (required by merge_asof)
    - For each Generated row, find the nearest subsequent Ended row
      matching on (room_id, event_type) within MAX_PAIRING_WINDOW_SEC
    - Deduplicate: each Ended row is consumed only once (first match wins)
    - Unpaired events (no matching Ended found) are kept with duration_sec = NaN

    Returns an event-level DataFrame sorted by start_time.
    """
    alarms = clean_df[clean_df['is_alarm']].copy()
    alarms = alarms.sort_values('timestamp').reset_index(drop=True)

    generated = alarms[alarms['event_state'] == 'Generated'].copy()
    ended     = alarms[alarms['event_state'] == 'Ended'].copy()

    events = _pair_events_fast(generated, ended)

    # Flag short-duration events as noise
    events['is_noise'] = events['duration_sec'] < NOISE_THRESHOLD_SEC

    events = events.sort_values('start_time').reset_index(drop=True)

    print(f"[Layer3] Paired events:          {len(events):,}")
    print(f"[Layer3] Noise (< {NOISE_THRESHOLD_SEC}s):       "
          f"{events['is_noise'].sum():,} ({events['is_noise'].mean()*100:.1f}%)")

    return events


def _pair_events_fast(generated: pd.DataFrame, ended: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized forward-matching using pandas merge_asof.

    merge_asof finds, for each Generated timestamp, the nearest Ended timestamp
    that is >= the Generated timestamp (direction='forward'), grouped by
    (room_id, event_type). This runs at C-level speed inside pandas.

    After merging, we:
    1. Enforce the MAX_PAIRING_WINDOW_SEC tolerance
    2. Deduplicate so each Ended row is only consumed once (first match wins)
    """
    # Prepare Generated side
    gen = generated.rename(columns={'timestamp': 'start_time'}).copy()
    gen = gen.sort_values('start_time').reset_index(drop=True)
    gen['_gen_idx'] = gen.index  # track original order for dedup

    # Prepare Ended side — only need timestamp + keys
    end = ended[['timestamp', 'room_id', 'event_type']].copy()
    end = end.rename(columns={'timestamp': 'end_time'})
    end = end.sort_values('end_time').reset_index(drop=True)
    end['_end_idx'] = end.index  # unique ID for dedup

    # ── merge_asof: forward match within tolerance ────────────────────────
    # For each Generated row, find the nearest Ended row with end_time >= start_time
    merged = pd.merge_asof(
        gen.sort_values('start_time'),
        end.sort_values('end_time'),
        left_on='start_time',
        right_on='end_time',
        by=['room_id', 'event_type'],
        direction='forward',
        tolerance=pd.Timedelta(seconds=MAX_PAIRING_WINDOW_SEC),
    )

    # ── Deduplicate: each Ended row can only pair once ────────────────────
    # When multiple Generated rows match the same Ended row, keep the earliest
    # Generated (smallest start_time), which is the closest temporal match.
    merged = merged.sort_values('start_time')
    has_match = merged['_end_idx'].notna()

    # Among rows with a match, drop duplicates on _end_idx keeping first
    matched   = merged[has_match].drop_duplicates(subset='_end_idx', keep='first')
    unmatched = merged[~has_match]

    # Rows that lost their match in dedup become unmatched
    deduped_gen_idx = set(matched['_gen_idx'].tolist()) | set(unmatched['_gen_idx'].tolist())
    lost = merged[~merged['_gen_idx'].isin(deduped_gen_idx)].copy()
    lost['end_time'] = pd.NaT
    lost['_end_idx'] = np.nan

    result = pd.concat([matched, unmatched, lost], ignore_index=True)
    result = result.sort_values('start_time').reset_index(drop=True)

    # ── Compute duration ──────────────────────────────────────────────────
    result['duration_sec'] = (
        result['end_time'] - result['start_time']
    ).dt.total_seconds()

    # ── Build output schema ───────────────────────────────────────────────
    output = result[[
        'room_id', 'start_time', 'end_time', 'duration_sec',
        'severity', 'severity_label', 'event_type', 'action_type',
        'metric_value', 'threshold', 'condition',
        'action_text', 'device_name', 'institution',
        'date', 'hour', 'weekday',
    ]].copy()

    output = output.rename(columns={'action_text': 'alert_text'})

    return output


def get_event_table_summary(events_df: pd.DataFrame) -> dict:
    """Return summary statistics for the Layer 3 output."""
    return {
        'total_events':      len(events_df),
        'unique_rooms':      events_df['room_id'].nunique(),
        'noise_count':       int(events_df['is_noise'].sum()),
        'noise_pct':         round(events_df['is_noise'].mean() * 100, 1),
        'mean_duration_sec': round(events_df['duration_sec'].mean(), 1),
        'median_duration_sec': round(events_df['duration_sec'].median(), 1),
        'critical_events':   int((events_df['severity'] == 3).sum()),
        'duration_by_type':  (
            events_df.groupby('event_type')['duration_sec']
            .agg(['count', 'mean', 'median'])
            .round(1)
            .to_dict('index')
        ),
    }