"""
Clinical Alert Intelligence Pipeline — Main Runner
Executes all 5 layers sequentially.

Usage:
    python run_pipeline.py --input data/your_file.csv
    python run_pipeline.py --input data/your_file.xlsx --output data/bi_output
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.layer1_ingest    import load_file, get_file_metadata
from pipeline.layer2_clean     import clean, get_clean_summary
from pipeline.layer3_events    import build_event_table, get_event_table_summary
from pipeline.layer4_analytics import (
    analyze_duration, analyze_frequency, analyze_noise,
    detect_escalation, detect_bursts, compute_time_between_events
)
from pipeline.layer5_bi_output import export_bi_datasets


def run_pipeline(input_path: str, output_dir: str = "./data/bi_output") -> dict:
    """
    Execute the full 5-layer Clinical Alert Intelligence Pipeline.
    Returns a dict containing all analytical outputs for downstream use.
    """
    print("=" * 60)
    print("  CLINICAL ALERT INTELLIGENCE PIPELINE")
    print("=" * 60)

    # ── Layer 1: Raw Ingestion ────────────────────────────────────────
    print("\n[1/5] Ingesting file...")
    raw_df = load_file(input_path)
    meta   = get_file_metadata(raw_df)
    print(f"  Rows:         {meta['total_rows']:,}")
    print(f"  Institutions: {', '.join(str(i) for i in meta['institutions'])}")
    print(f"  Beds/Rooms:   {meta['beds']}")

    # ── Layer 2: Clean & Parse ────────────────────────────────────────
    print("\n[2/5] Cleaning & parsing...")
    clean_df = clean(raw_df)
    summary  = get_clean_summary(clean_df)
    print(f"  Total alarms:      {summary['total_alarms']:,}")
    print(f"  Rooms:             {summary['rooms']}")
    print(f"  Critical (★★★):    {summary['critical_alarms']:,}")

    # ── Layer 3: Event-Level Table ────────────────────────────────────
    print("\n[3/5] Building event-level table (pairing Generated ↔ Ended)...")
    events_df = build_event_table(clean_df)
    events_df = compute_time_between_events(events_df)

    # ── Layer 4: Feature Engineering ─────────────────────────────────
    print("\n[4/5] Running feature engineering & analytics...")
    duration_stats          = analyze_duration(events_df)
    frequency_stats         = analyze_frequency(events_df)
    noise_stats             = analyze_noise(events_df)
    escalation_df, pre_crit = detect_escalation(events_df)
    bursts_df               = detect_bursts(events_df)

    print(f"\n  Duration:   median={duration_stats['overall']['median_sec']}s, "
          f"mean={duration_stats['overall']['mean_sec']}s")
    print(f"  Noise:      {noise_stats['overall_noise_pct']}% of alarms < "
          f"{noise_stats['noise_threshold_used']}s")
    print(f"  Escalations: {len(escalation_df):,} sequences detected")
    print(f"  Bursts:      {len(bursts_df):,} clusters detected")

    # ── Layer 5: BI Export ────────────────────────────────────────────
    print("\n[5/5] Exporting Power BI datasets...")
    bi_outputs = export_bi_datasets(events_df, escalation_df, bursts_df, output_dir)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Output directory: {Path(output_dir).resolve()}")
    print("=" * 60)

    return {
        'raw_df':          raw_df,
        'clean_df':        clean_df,
        'events_df':       events_df,
        'escalation_df':   escalation_df,
        'pre_critical_df': pre_crit,
        'bursts_df':       bursts_df,
        'duration_stats':  duration_stats,
        'frequency_stats': frequency_stats,
        'noise_stats':     noise_stats,
        'bi_outputs':      bi_outputs,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clinical Alert Intelligence Pipeline — PIC iX Monitor Export Analyzer'
    )
    parser.add_argument('--input',  required=True,               help='Path to CSV or Excel file')
    parser.add_argument('--output', default='./data/bi_output',  help='Output directory for BI files')
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
