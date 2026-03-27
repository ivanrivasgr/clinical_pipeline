"""
Layer 5 — BI Output (Power BI Ready)
Exports final datasets optimized for direct import into Power BI.

Historical accumulation:
  Each pipeline run saves to a date-stamped subfolder under bi_output/.
  Power BI connects to the root bi_output/ folder using "Get Data → Folder"
  and combines all historical CSVs automatically.

  bi_output/
  ├── 2025-07_Jul/
  │   ├── fact_alerts.csv
  │   ├── dim_rooms.csv
  │   └── ...
  ├── 2025-08_Aug/
  │   ├── fact_alerts.csv
  │   └── ...
  └── _combined/
      ├── fact_alerts.csv       ← all months merged
      ├── dim_rooms.csv         ← all months merged (deduplicated)
      └── ...

Output files (per period):
  fact_alerts.csv             Main event-level fact table
  dim_rooms.csv               Room dimension
  dim_event_types.csv         Event type dimension with clinical metadata
  agg_room_day.csv            Pre-aggregated room × day summary
  agg_hourly.csv              Pre-aggregated hourly summary
  escalation_sequences.csv    Escalation patterns
  bursts.csv                  Alert burst patterns

Power BI Setup (one-time):
  1. Home → Get Data → Folder
  2. Point to bi_output/_combined/
  3. Power BI will read all CSV files
  4. Each new pipeline run updates _combined/ automatically
  5. Just click Refresh in Power BI to pull new data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def export_bi_datasets(
    events_df:      pd.DataFrame,
    escalation_df:  pd.DataFrame,
    bursts_df:      pd.DataFrame,
    output_dir:     str = "./data/bi_output",
    period_label:   Optional[str] = None,
) -> dict:
    """
    Main export function. Creates all Power BI-ready output tables.

    Historical accumulation:
      - Saves current run to a date-stamped subfolder (e.g. 2025-07_Jul/)
      - Rebuilds _combined/ folder by merging all historical subfolders
      - Power BI points to _combined/ and just clicks Refresh

    Args:
        events_df:     Paired event data from Layer 3/4
        escalation_df: Escalation sequences from Layer 4
        bursts_df:     Alert bursts from Layer 4
        output_dir:    Root output directory (default: ./data/bi_output)
        period_label:  Optional override label (e.g. "2025-07_Jul").
                       If None, auto-detected from date range in the data.

    Returns:
        dict mapping {table_name: DataFrame}
    """
    root_path = Path(output_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    # ── Determine period label from data ──────────────────────────────────
    if period_label is None:
        period_label = _detect_period_label(events_df)

    period_path = root_path / period_label
    period_path.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # ── 1. Central fact table ─────────────────────────────────────────────
    fact = _build_fact_table(events_df)
    fact['export_period'] = period_label
    _save_csv(fact, period_path / "fact_alerts.csv")
    outputs['fact_alerts'] = fact

    # ── 2. Dimension tables ───────────────────────────────────────────────
    dim_rooms = _build_dim_rooms(events_df)
    dim_rooms['export_period'] = period_label
    _save_csv(dim_rooms, period_path / "dim_rooms.csv")
    outputs['dim_rooms'] = dim_rooms

    dim_events = _build_dim_event_types(events_df)
    dim_events['export_period'] = period_label
    _save_csv(dim_events, period_path / "dim_event_types.csv")
    outputs['dim_event_types'] = dim_events

    # ── 3. Pre-aggregated tables ──────────────────────────────────────────
    agg_room_day = _build_agg_room_day(events_df)
    agg_room_day['export_period'] = period_label
    _save_csv(agg_room_day, period_path / "agg_room_day.csv")
    outputs['agg_room_day'] = agg_room_day

    agg_hourly = _build_agg_hourly(events_df)
    agg_hourly['export_period'] = period_label
    _save_csv(agg_hourly, period_path / "agg_hourly.csv")
    outputs['agg_hourly'] = agg_hourly

    # ── 4. Pattern analysis tables ────────────────────────────────────────
    if not escalation_df.empty:
        esc = escalation_df.copy()
        esc['export_period'] = period_label
        _save_csv(esc, period_path / "escalation_sequences.csv")
        outputs['escalation_sequences'] = esc

    if not bursts_df.empty:
        bst = bursts_df.copy()
        bst['export_period'] = period_label
        _save_csv(bst, period_path / "bursts.csv")
        outputs['bursts'] = bst

    # ── 5. Rebuild combined folder ────────────────────────────────────────
    _rebuild_combined(root_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[Layer5] BI export complete → {period_path.resolve()}")
    print(f"[Layer5] Period: {period_label}")
    for name, df in outputs.items():
        print(f"  {name}: {len(df):,} rows × {len(df.columns)} cols")

    # List all historical periods
    periods = _list_periods(root_path)
    print(f"\n[Layer5] Historical periods on disk: {len(periods)}")
    for p in periods:
        print(f"  └── {p}")

    return outputs


def _detect_period_label(events_df: pd.DataFrame) -> str:
    """Auto-detect period label from the date range in the data."""
    try:
        dates = pd.to_datetime(events_df['date'])
        min_date = dates.min()
        max_date = dates.max()

        if min_date.month == max_date.month and min_date.year == max_date.year:
            # Single month: "2025-07_Jul"
            return min_date.strftime("%Y-%m_%b")
        else:
            # Spans multiple months: "2025-07_to_2025-08"
            return f"{min_date.strftime('%Y-%m')}_to_{max_date.strftime('%Y-%m')}"
    except Exception:
        # Fallback: use today's date
        return pd.Timestamp.now().strftime("%Y-%m_%b")


def _list_periods(root_path: Path) -> list:
    """List all historical period subfolders."""
    return sorted([
        d.name for d in root_path.iterdir()
        if d.is_dir() and d.name != "_combined"
    ])


def _rebuild_combined(root_path: Path):
    """
    Merge all historical period subfolders into _combined/.
    Power BI points here and just clicks Refresh.
    """
    combined_path = root_path / "_combined"
    combined_path.mkdir(parents=True, exist_ok=True)

    periods = _list_periods(root_path)
    if not periods:
        return

    # Tables to combine
    table_names = [
        "fact_alerts", "dim_rooms", "dim_event_types",
        "agg_room_day", "agg_hourly",
        "escalation_sequences", "bursts",
    ]

    for table in table_names:
        frames = []
        for period in periods:
            csv_path = root_path / period / f"{table}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    frames.append(df)
                except Exception:
                    pass

        if frames:
            combined = pd.concat(frames, ignore_index=True)

            # Deduplicate dimension tables (keep latest period's version)
            if table == "dim_rooms":
                combined = combined.drop_duplicates(subset=['room_id'], keep='last')
            elif table == "dim_event_types":
                combined = combined.drop_duplicates(subset=['event_type'], keep='last')

            _save_csv(combined, combined_path / f"{table}.csv")
            print(f"  [combined] {table}: {len(combined):,} rows ({len(frames)} periods)")


def _build_fact_table(events: pd.DataFrame) -> pd.DataFrame:
    """Build the central fact table with proper dtypes for Power BI."""
    fact = events[[
        'room_id', 'start_time', 'end_time', 'duration_sec',
        'severity', 'severity_label', 'event_type', 'action_type',
        'metric_value', 'threshold', 'condition',
        'alert_text', 'device_name', 'institution',
        'date', 'hour', 'weekday', 'is_noise',
    ]].copy()

    # Enforce correct types
    fact['duration_sec']  = pd.to_numeric(fact['duration_sec'],  errors='coerce').round(1)
    fact['metric_value']  = pd.to_numeric(fact['metric_value'],  errors='coerce').round(1)
    fact['threshold']     = pd.to_numeric(fact['threshold'],     errors='coerce').round(1)
    fact['severity']      = fact['severity'].astype('Int8')
    fact['hour']          = fact['hour'].astype('Int8')
    fact['is_noise']      = fact['is_noise'].astype(bool)
    fact['date']          = pd.to_datetime(fact['date'])

    # Derived columns for Power BI measures
    fact['duration_min']          = (fact['duration_sec'] / 60).round(2)
    fact['noise_flag']            = fact['is_noise'].map({True: 'Noise', False: 'Signal'})
    fact['severity_text']         = fact['severity'].map({1: 'Low (★)', 2: 'Medium (★★)', 3: 'Critical (★★★)'})
    fact['delta_from_threshold']  = (fact['metric_value'] - fact['threshold']).round(1)

    return fact.sort_values('start_time').reset_index(drop=True)


def _build_dim_rooms(events: pd.DataFrame) -> pd.DataFrame:
    """Room dimension table with aggregate statistics."""
    dim = events.groupby('room_id').agg(
        institution    = ('institution', 'first'),
        device_name    = ('device_name', 'first'),
        total_alerts   = ('event_type', 'count'),
        critical_alerts = ('severity', lambda x: (x == 3).sum()),
        noise_alerts   = ('is_noise', 'sum'),
        first_seen     = ('start_time', 'min'),
        last_seen      = ('start_time', 'max'),
    ).reset_index()
    dim['noise_pct'] = (dim['noise_alerts'] / dim['total_alerts'] * 100).round(1)
    return dim


def _build_dim_event_types(events: pd.DataFrame) -> pd.DataFrame:
    """Event type dimension with clinical category and risk level metadata."""
    dim = events.groupby('event_type').agg(
        count            = ('event_type', 'count'),
        avg_severity     = ('severity', 'mean'),
        avg_duration_sec = ('duration_sec', 'mean'),
        noise_pct        = ('is_noise', lambda x: round(x.mean() * 100, 1)),
    ).reset_index().round(2)

    # Clinical category mapping
    category_map = {
        'HR':          'Cardiac',   'Brady':         'Cardiac',
        'VTach':       'Cardiac',   'VFib':          'Cardiac',
        'NonSustainVT':'Cardiac',   'IrregularHR':   'Cardiac',
        'RunPVCs':     'Cardiac',   'Asystole':      'Cardiac',
        'SpO2':        'Respiratory', 'Desat':        'Respiratory',
        'RR':          'Respiratory', 'Apnea':        'Respiratory',
        'NBPs':        'Hemodynamic', 'NBPd':         'Hemodynamic',
        'TEMP':        'Metabolic',
    }
    dim['clinical_category'] = dim['event_type'].map(category_map).fillna('Other')

    # Clinical risk level
    high_risk   = {'VTach', 'VFib', 'Asystole', 'Desat', 'Apnea'}
    medium_risk = {'Brady', 'IrregularHR', 'NonSustainVT'}
    dim['risk_level'] = dim['event_type'].apply(
        lambda x: 'High' if x in high_risk else ('Medium' if x in medium_risk else 'Low')
    )

    return dim.sort_values('count', ascending=False)


def _build_agg_room_day(events: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregated room × day table for Power BI dashboard visuals."""
    agg = events.groupby(['room_id', 'date']).agg(
        total_alerts        = ('event_type', 'count'),
        critical_alerts     = ('severity', lambda x: (x == 3).sum()),
        noise_alerts        = ('is_noise', 'sum'),
        unique_event_types  = ('event_type', 'nunique'),
        avg_duration_sec    = ('duration_sec', 'mean'),
        max_severity        = ('severity', 'max'),
    ).reset_index()
    agg['date']             = pd.to_datetime(agg['date'])
    agg['noise_pct']        = (agg['noise_alerts'] / agg['total_alerts'] * 100).round(1)
    agg['avg_duration_sec'] = agg['avg_duration_sec'].round(1)
    return agg


def _build_agg_hourly(events: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregated hourly breakdown for Power BI time-of-day analysis."""
    agg = events.groupby(['date', 'hour']).agg(
        total_alerts    = ('event_type', 'count'),
        critical_alerts = ('severity', lambda x: (x == 3).sum()),
        noise_alerts    = ('is_noise', 'sum'),
        unique_rooms    = ('room_id', 'nunique'),
    ).reset_index()
    agg['date']       = pd.to_datetime(agg['date'])
    agg['hour_label'] = agg['hour'].apply(lambda h: f"{h:02d}:00")
    return agg


def _save_csv(df: pd.DataFrame, csv_path: Path):
    """Save as CSV. Simple, universal, Power BI compatible."""
    df.to_csv(csv_path, index=False)