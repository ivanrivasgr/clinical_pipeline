"""
Layer 3 — Event-Level Dataset (Core)
Pairs Generated ↔ Ended events to compute duration_sec.
This is the primary analytical table used by all downstream layers.

Output schema:
    room_id, start_time, end_time, duration_sec, severity, event_type,
    metric_value, threshold, condition, alert_text, device_name,
    institution, date, hour, weekday, severity_label, is_noise
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

    Pairing strategy:
    - Filter to clinical alarms only (is_alarm == True)
    - Split into 'Generated' and 'Ended' subsets
    - For each Generated row, find the nearest subsequent Ended row
      matching on (room_id, event_type) within MAX_PAIRING_WINDOW_SEC
    - Unpaired events (no matching Ended found) are kept with duration_sec = NaN

    Returns an event-level DataFrame sorted by start_time.
    """
    alarms = clean_df[clean_df['is_alarm']].copy()
    alarms = alarms.sort_values('timestamp').reset_index(drop=True)

    generated = alarms[alarms['event_state'] == 'Generated'].copy()
    ended     = alarms[alarms['event_state'] == 'Ended'].copy()

    events = _pair_events(generated, ended)

    # Flag short-duration events as noise
    events['is_noise'] = events['duration_sec'] < NOISE_THRESHOLD_SEC

    events = events.sort_values('start_time').reset_index(drop=True)

    print(f"[Layer3] Paired events:          {len(events):,}")
    print(f"[Layer3] Noise (< {NOISE_THRESHOLD_SEC}s):       "
          f"{events['is_noise'].sum():,} ({events['is_noise'].mean()*100:.1f}%)")

    return events


def _pair_events(generated: pd.DataFrame, ended: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy forward-matching:
    For each Generated row, find the earliest matching Ended row
    within the pairing window. Each Ended row can only be used once.
    """
    rows = []
    used_ended_idx = set()

    for _, gen in generated.iterrows():
        duration_sec = np.nan
        end_time     = pd.NaT

        # Find candidate Ended events for this room + event_type
        candidates = ended[
            (ended['room_id']    == gen['room_id']) &
            (ended['event_type'] == gen['event_type']) &
            (ended['timestamp']  >= gen['timestamp']) &
            (ended['timestamp']  <= gen['timestamp'] + pd.Timedelta(seconds=MAX_PAIRING_WINDOW_SEC)) &
            (~ended.index.isin(used_ended_idx))
        ].sort_values('timestamp')

        if not candidates.empty:
            best = candidates.iloc[0]
            used_ended_idx.add(best.name)
            duration_sec = (best['timestamp'] - gen['timestamp']).total_seconds()
            end_time     = best['timestamp']

        rows.append({
            'room_id':        gen['room_id'],
            'start_time':     gen['timestamp'],
            'end_time':       end_time,
            'duration_sec':   duration_sec,
            'severity':       gen['severity'],
            'severity_label': gen['severity_label'],
            'event_type':     gen['event_type'],
            'metric_value':   gen['metric_value'],
            'threshold':      gen['threshold'],
            'condition':      gen['condition'],
            'alert_text':     gen['action_text'],
            'device_name':    gen['device_name'],
            'institution':    gen['institution'],
            'date':           gen['date'],
            'hour':           gen['hour'],
            'weekday':        gen['weekday'],
            'action_type':    gen['action_type'],
        })

    return pd.DataFrame(rows)


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
