# 🫀 Clinical Alert Intelligence Pipeline

**Driscoll Children's Hospital — RGV Clinical Informatics**

A 5-layer Python pipeline that processes PIC iX bedside monitor alarm exports and transforms raw, unstructured alert data into clinically actionable insights and Power BI-ready datasets.

---

## Overview

Hospital bedside monitors generate thousands of alarms daily — the vast majority are clinically insignificant noise that contributes to **alert fatigue**, a leading patient safety concern. This pipeline ingests raw PIC iX monitor exports, pairs alarm events, detects noise, identifies escalation patterns, and outputs structured datasets ready for Power BI dashboards.

### Key Metrics (from sample dataset)
| Metric | Value |
|--------|-------|
| Raw alarm events | 49,586 |
| Paired events | 24,793 |
| Noise rate | 46.4% |
| Critical alarms (★★★) | 1,935 |
| Escalation sequences | 586 |

---

## Pipeline Architecture

```
Layer 1 — Raw Ingestion      Load CSV/Excel from PIC iX export
Layer 2 — Clean & Parse      Extract severity, event type, metric, threshold
Layer 3 — Event Pairing      Pair Generated ↔ Ended events, compute duration
Layer 4 — Analytics          Duration, frequency, noise, escalation, bursts
Layer 5 — BI Output          Power BI-ready Parquet + CSV star-schema datasets
```

### Star Schema Output
```
fact_alerts
├── room_id ──────────► dim_rooms
├── event_type ────────► dim_event_types
├── date ──────────────► dim_date (auto Power BI)
└── measures
     ├── duration_sec
     ├── severity
     ├── is_noise
     └── metric_value

agg_room_day          Pre-aggregated room × day
agg_hourly            Pre-aggregated hourly heatmaps
escalation_sequences  Severity progression chains
bursts                Burst detection (≥3 events in 5 min)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/clinical-alert-intelligence.git
cd clinical-alert-intelligence
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run streamlit_app/app.py
```

Then open `http://localhost:8501` in your browser, upload a PIC iX export file, and click **Execute Pipeline**.

---

## Project Structure

```
clinical-alert-intelligence/
├── pipeline/
│   ├── __init__.py
│   ├── layer1_ingest.py        # CSV/Excel loader
│   ├── layer2_clean.py         # Parsing & cleaning
│   ├── layer3_events.py        # Event pairing (Generated ↔ Ended)
│   ├── layer4_analytics.py     # Duration, frequency, noise, escalation, bursts
│   └── layer5_bi_output.py     # Star-schema BI export
├── streamlit_app/
│   └── app.py                  # Dashboard UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Power BI Integration

The pipeline outputs CSV files in a star-schema format designed for direct import into Power BI:

1. Open Power BI Desktop → **Get Data** → **CSV**
2. Import `fact_alerts.csv` as the central fact table
3. Import `dim_rooms.csv` and `dim_event_types.csv` as dimension tables
4. Create relationships via `room_id` and `event_type`
5. Use `agg_room_day.csv` and `agg_hourly.csv` for pre-aggregated visuals

For automated refresh, configure a scheduled pipeline run that outputs to a shared location (SharePoint, OneDrive, or a network drive) connected to Power BI Gateway.

---

## License

Internal use — Driscoll Children's Hospital.
