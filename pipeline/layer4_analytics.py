"""
Layer 4 — Feature Engineering & Analytics
Generates all analytical features from the event-level table (Layer 3).

Modules:
  1. Duration Analysis        — mean, median, p75, p95 by event type and severity
  2. Frequency Analysis       — alarms per room, hour, weekday, event type
  3. Noise Detection          — short-duration alarms with no clinical value
  4. Escalation Detection     — severity progression (★ → ★★ → ★★★)
  5. Temporal Patterns        — alert bursts, risk windows
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ─── Configuration ────────────────────────────────────────────────────────────

NOISE_THRESHOLD_SEC    = 10   # Alarms shorter than 10s are considered noise
BURST_GAP_SEC          = 300  # Events within 5 min of each other form a burst
ESCALATION_WINDOW_MIN  = 60   # Look back 60 min when detecting escalation sequences
PRE_CRITICAL_WINDOW_MIN = 30  # Look back 30 min before a severity-3 event


# ─── 1. Duration Analysis ─────────────────────────────────────────────────────

def analyze_duration(events: pd.DataFrame) -> dict:
    """
    Compute duration statistics overall and per event type.
    Returns a dict with 'overall', 'by_type', and 'by_severity' DataFrames.
    """
    df = events.dropna(subset=['duration_sec'])

    overall = {
        'mean_sec':    round(df['duration_sec'].mean(), 1),
        'median_sec':  round(df['duration_sec'].median(), 1),
        'p75_sec':     round(df['duration_sec'].quantile(0.75), 1),
        'p95_sec':     round(df['duration_sec'].quantile(0.95), 1),
        'max_sec':     round(df['duration_sec'].max(), 1),
        'noise_count': int((df['duration_sec'] < NOISE_THRESHOLD_SEC).sum()),
        'noise_pct':   round((df['duration_sec'] < NOISE_THRESHOLD_SEC).mean() * 100, 1),
    }

    by_type = (
        df.groupby('event_type')['duration_sec']
        .agg(
            count     = 'count',
            mean      = lambda x: round(x.mean(), 1),
            median    = lambda x: round(x.median(), 1),
            noise_pct = lambda x: round((x < NOISE_THRESHOLD_SEC).mean() * 100, 1),
        )
        .reset_index()
        .sort_values('count', ascending=False)
    )

    by_severity = (
        df.groupby('severity')['duration_sec']
        .agg(count='count', mean=lambda x: round(x.mean(), 1), median='median')
        .reset_index()
    )

    return {
        'overall':     overall,
        'by_type':     by_type,
        'by_severity': by_severity,
    }


# ─── 2. Frequency Analysis ───────────────────────────────────────────────────

def analyze_frequency(events: pd.DataFrame) -> dict:
    """
    Return alarm frequency breakdowns by room, hour, weekday, and event type.
    """
    by_room = (
        events.groupby('room_id')
        .agg(
            total_alerts   = ('event_type', 'count'),
            critical_alerts = ('severity', lambda x: (x == 3).sum()),
            noise_alerts   = ('is_noise', 'sum'),
        )
        .reset_index()
        .sort_values('total_alerts', ascending=False)
    )

    by_hour = (
        events.groupby('hour')
        .agg(
            total_alerts    = ('event_type', 'count'),
            critical_alerts = ('severity', lambda x: (x == 3).sum()),
        )
        .reset_index()
        .sort_values('hour')
    )

    by_type = (
        events.groupby('event_type')
        .agg(
            count        = ('event_type', 'count'),
            severity_avg = ('severity', 'mean'),
            noise_pct    = ('is_noise', lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .sort_values('count', ascending=False)
    )

    by_room_day = (
        events.groupby(['room_id', 'date'])
        .size()
        .reset_index(name='alert_count')
    )

    by_weekday = (
        events.groupby('weekday')
        .agg(total_alerts=('event_type', 'count'))
        .reset_index()
    )
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_weekday['weekday'] = pd.Categorical(
        by_weekday['weekday'], categories=day_order, ordered=True
    )
    by_weekday = by_weekday.sort_values('weekday')

    return {
        'by_room':     by_room,
        'by_hour':     by_hour,
        'by_type':     by_type,
        'by_room_day': by_room_day,
        'by_weekday':  by_weekday,
    }


# ─── 3. Noise Detection ──────────────────────────────────────────────────────

def analyze_noise(events: pd.DataFrame) -> dict:
    """
    Noise = clinical alarms with duration < NOISE_THRESHOLD_SEC.
    These alarms fire and clear so quickly they provide no actionable clinical value.
    Returns overall noise stats and a per-type breakdown.
    """
    df = events.dropna(subset=['duration_sec'])

    noise  = df[df['is_noise']]
    signal = df[~df['is_noise']]

    noise_by_type = (
        df.groupby('event_type')
        .agg(
            total       = ('is_noise', 'count'),
            noise_count = ('is_noise', 'sum'),
            noise_pct   = ('is_noise', lambda x: round(x.mean() * 100, 1)),
            avg_duration = ('duration_sec', lambda x: round(x.mean(), 1)),
        )
        .reset_index()
        .sort_values('noise_pct', ascending=False)
    )

    # Threshold impact simulation:
    # Identify "marginal" alarms — events within 5% of threshold value.
    # These are the most likely candidates for threshold adjustment.
    threshold_sim = []
    for event_type in df['event_type'].unique():
        sub = df[
            (df['event_type'] == event_type) &
            df['metric_value'].notna() &
            df['threshold'].notna()
        ]
        if len(sub) < 5:
            continue
        current_noise = sub['is_noise'].mean() * 100
        close = sub[
            abs(sub['metric_value'] - sub['threshold']) / sub['threshold'].clip(1) < 0.05
        ]
        threshold_sim.append({
            'event_type':       event_type,
            'total':            len(sub),
            'current_noise_pct': round(current_noise, 1),
            'marginal_alarms':  len(close),
            'marginal_pct':     round(len(close) / len(sub) * 100, 1),
        })

    return {
        'total_noise':          len(noise),
        'total_signal':         len(signal),
        'overall_noise_pct':    round(len(noise) / len(df) * 100, 1),
        'noise_by_type':        noise_by_type,
        'threshold_simulation': pd.DataFrame(threshold_sim),
        'noise_threshold_used': NOISE_THRESHOLD_SEC,
    }


# ─── 4. Escalation Detection ─────────────────────────────────────────────────

def detect_escalation(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect escalation patterns: sequences where severity increases
    within ESCALATION_WINDOW_MIN for the same room.

    Returns:
        escalation_sequences — room/time/severity transitions
        pre_critical_context — events in the 30 min before each severity-3 alarm
    """
    df = events.sort_values(['room_id', 'start_time']).copy()

    # ── Build escalation sequences
    sequences = []
    for room, room_df in df.groupby('room_id'):
        room_df = room_df.sort_values('start_time').reset_index(drop=True)
        for i, row in room_df.iterrows():
            window_start = row['start_time'] - pd.Timedelta(minutes=ESCALATION_WINDOW_MIN)
            prior = room_df[
                (room_df['start_time'] >= window_start) &
                (room_df['start_time'] <  row['start_time'])
            ]
            if prior.empty:
                continue
            max_prior_sev = prior['severity'].max()
            if row['severity'] > max_prior_sev:
                sequences.append({
                    'room_id':           room,
                    'escalation_time':   row['start_time'],
                    'from_severity':     int(max_prior_sev),
                    'to_severity':       int(row['severity']),
                    'trigger_event':     row['event_type'],
                    'prior_events_count': len(prior),
                    'prior_event_types': ', '.join(prior['event_type'].unique()),
                })

    escalation_df = pd.DataFrame(sequences)

    # ── Build pre-critical context (30 min window before each severity-3 event)
    critical_events  = df[df['severity'] == 3].copy()
    pre_critical_rows = []
    for _, crit in critical_events.iterrows():
        window_start = crit['start_time'] - pd.Timedelta(minutes=PRE_CRITICAL_WINDOW_MIN)
        prior = df[
            (df['room_id']    == crit['room_id']) &
            (df['start_time'] >= window_start) &
            (df['start_time'] <  crit['start_time'])
        ].copy()
        prior['critical_event_type'] = crit['event_type']
        prior['critical_event_time'] = crit['start_time']
        prior['minutes_before_critical'] = (
            (crit['start_time'] - prior['start_time']).dt.total_seconds() / 60
        ).round(1)
        pre_critical_rows.append(prior)

    pre_critical_df = (
        pd.concat(pre_critical_rows, ignore_index=True)
        if pre_critical_rows else pd.DataFrame()
    )

    print(f"[Layer4] Escalation sequences detected: {len(escalation_df):,}")
    print(f"[Layer4] Critical events (★★★):         {len(critical_events):,}")

    return escalation_df, pre_critical_df


# ─── 5. Temporal Patterns ────────────────────────────────────────────────────

def detect_bursts(events: pd.DataFrame) -> pd.DataFrame:
    """
    Detect alert bursts: clusters of 3+ consecutive events per room
    with less than BURST_GAP_SEC between them.

    Returns a DataFrame with burst_id, room_id, burst_start/end,
    event_count, severity_max, event_types, has_critical.
    """
    df = events.sort_values(['room_id', 'start_time']).copy()
    bursts   = []
    burst_id = 0

    for room, room_df in df.groupby('room_id'):
        room_df = room_df.sort_values('start_time').reset_index(drop=True)
        times   = room_df['start_time'].values
        n       = len(times)
        i       = 0

        while i < n:
            j = i + 1
            while j < n:
                gap = (pd.Timestamp(times[j]) - pd.Timestamp(times[j - 1])).total_seconds()
                if gap <= BURST_GAP_SEC:
                    j += 1
                else:
                    break

            burst_len = j - i
            if burst_len >= 3:
                burst_rows = room_df.iloc[i:j]
                bursts.append({
                    'burst_id':    burst_id,
                    'room_id':     room,
                    'burst_start': burst_rows['start_time'].min(),
                    'burst_end':   burst_rows['start_time'].max(),
                    'duration_min': round(
                        (burst_rows['start_time'].max() - burst_rows['start_time'].min())
                        .total_seconds() / 60, 1
                    ),
                    'event_count':  burst_len,
                    'severity_max': int(burst_rows['severity'].max()),
                    'event_types':  ', '.join(burst_rows['event_type'].unique()),
                    'has_critical': int((burst_rows['severity'] == 3).any()),
                })
                burst_id += 1
            i = j

    print(f"[Layer4] Alert bursts detected: {burst_id}")
    return pd.DataFrame(bursts)


def compute_time_between_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'seconds_since_last_event' column computed per room.
    Useful for detecting periods of rapid sequential alarming.
    """
    df = events.sort_values(['room_id', 'start_time']).copy()
    df['seconds_since_last_event'] = (
        df.groupby('room_id')['start_time']
        .diff()
        .dt.total_seconds()
    )
    return df


def get_risk_windows(events: pd.DataFrame, window_hours: int = 4) -> pd.DataFrame:
    """
    Identify high-activity time windows by resampling events into
    rolling N-hour buckets per room.

    Returns the top windows by alarm count, including max severity and event types.
    """
    df = events.copy()
    df = df.set_index('start_time').sort_index()

    results = []
    for room, room_df in df.groupby('room_id'):
        if len(room_df) < 5:
            continue
        rolling = room_df.resample(f'{window_hours}H').agg(
            count        = ('event_type', 'count'),
            max_severity = ('severity', 'max'),
            event_types  = ('event_type', lambda x: ', '.join(x.unique())),
        ).reset_index()
        rolling['room_id'] = room
        results.append(rolling)

    if not results:
        return pd.DataFrame()

    risk_df = pd.concat(results, ignore_index=True)
    return risk_df.sort_values('count', ascending=False)
