# Windhoek Traffic Congestion Analysis

An end-to-end data pipeline that fetches Windhoek's road network from OpenStreetMap, models peak-hour congestion patterns, and renders an interactive choropleth map.

![Congestion Distribution](outputs/congestion_distribution.png)

## Overview

This project analyses traffic congestion across Windhoek's arterial road network by:

1. **Fetching** the full drivable road graph from OpenStreetMap via OSMnx
2. **Simulating** AM peak (07:00–09:00) and PM peak (16:00–18:00) GPS speed observations relative to posted speed limits
3. **Scoring** each road segment with a congestion index (0 = free flow → 1 = standstill)
4. **Visualising** results on an interactive Folium map with per-segment popups

> **Note:** Speed observations are currently simulated using road-type-weighted random sampling. The pipeline is designed so real probe-vehicle CSV data can be plugged in at Step 2 with no changes to the rest of the code.

## Project Structure

```
windhoek-traffic/
├── src/
│   └── pipeline.py          # Main pipeline (fetch → simulate → score → map)
├── notebooks/
│   └── analysis.ipynb       # Exploratory analysis & charts
├── data/                    # Auto-generated: road_edges.gpkg, congestion_scores.csv
├── outputs/                 # Auto-generated: HTML map, PNG charts
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/Genuine17/Genuine17.git
cd windhoek-traffic
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline
python src/pipeline.py

# 3. Open the map
open outputs/windhoek_congestion_map.html

# 4. Or explore the notebook
jupyter notebook notebooks/analysis.ipynb
```

## Output

| File | Description |
|------|-------------|
| `outputs/windhoek_congestion_map.html` | Interactive map — click any road for details |
| `outputs/congestion_distribution.png` | Score histogram + severity bar chart |
| `outputs/am_vs_pm_by_road_type.png` | AM vs PM comparison by road category |
| `data/congestion_scores.csv` | Per-segment congestion scores (exportable) |

## Congestion Scoring

| Score | Label | Meaning |
|-------|-------|---------|
| 0.0 – 0.2 | Free flow | Traffic moving at or near speed limit |
| 0.2 – 0.4 | Light | Minor slowdowns |
| 0.4 – 0.6 | Moderate | Noticeable delays |
| 0.6 – 0.8 | Heavy | Significant congestion |
| 0.8 – 1.0 | Severe | Near standstill |

## Extending with Real Data

Replace `simulate_gps_observations()` in `src/pipeline.py` with a loader for your own GPS/probe data. The function must return a DataFrame with `speed_am` and `speed_pm` columns (km/h) aligned to OSM edge IDs.

## Tools & Libraries

- [OSMnx](https://osmnx.readthedocs.io/) — road network download & graph processing
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data pipeline
- [Folium](https://python-visualization.github.io/folium/) — interactive map rendering
- [Matplotlib](https://matplotlib.org/) — static charts

## Author

**Phillann G.T Shitaleni** — BSc Informatics, Namibia University of Science and Technology  
[github.com/Genuine17](https://github.com/Genuine17/Genuine17)
