"""
Windhoek Traffic Congestion Analysis Pipeline
==============================================
Fetches road network data from OpenStreetMap, simulates peak-hour
GPS speed observations, computes congestion scores per road segment,
and exports an interactive Folium choropleth map.
"""

import osmnx as ox
import pandas as pd
import numpy as np
import folium
import json
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CITY        = "Windhoek, Namibia"
OUTPUT_DIR  = Path(__file__).parent.parent / "outputs"
DATA_DIR    = Path(__file__).parent.parent / "data"
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ── Step 1: Fetch road network from OSM ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_road_network(city: str = CITY):
    """Download drivable road network for Windhoek via OSMnx."""
    print(f"[1/4] Fetching road network for {city}...")
    G = ox.graph_from_place(city, network_type="drive")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # Save raw edges for reproducibility
    gdf_edges.to_file(DATA_DIR / "road_edges.gpkg", driver="GPKG")
    print(f"      {len(gdf_edges)} road segments loaded.")
    return gdf_edges


# ── Step 2: Simulate peak-hour GPS speed observations ────────────────────────
@st.cache_data(show_spinner=False)
def simulate_gps_observations(gdf_edges: pd.DataFrame, seed: int = RANDOM_SEED):
    """
    Simulate observed speeds for AM peak (07:00–09:00) and PM peak (16:00–18:00).
    In a real deployment this would be replaced with actual probe-vehicle CSV data.
    Uses road speed limits from OSM (maxspeed tag) as free-flow reference.
    """
    print("[2/4] Simulating GPS speed observations...")
    rng = np.random.default_rng(seed)

    edges = gdf_edges.copy()

    # Parse speed limit; default to 60 km/h where missing
    edges["speed_limit"] = (
        edges["maxspeed"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
        .fillna(60)
        .clip(upper=120)
    )

    # Congestion factor: arterials slow more in peaks
    is_arterial = edges["highway"].astype(str).str.contains(
        "primary|secondary|trunk", na=False
    )

    # AM peak speeds (heavier congestion on arterials)
    edges["speed_am"] = np.where(
        is_arterial,
        edges["speed_limit"] * rng.uniform(0.35, 0.65, len(edges)),
        edges["speed_limit"] * rng.uniform(0.55, 0.85, len(edges)),
    )

    # PM peak speeds (slightly lighter)
    edges["speed_pm"] = np.where(
        is_arterial,
        edges["speed_limit"] * rng.uniform(0.40, 0.70, len(edges)),
        edges["speed_limit"] * rng.uniform(0.60, 0.90, len(edges)),
    )

    edges["speed_freeflow"] = edges["speed_limit"]

    return edges


# ── Step 3: Compute congestion score ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_congestion(edges: pd.DataFrame):
    """
    Congestion score = 1 - (observed_speed / free_flow_speed).
    0 = free flow, 1 = standstill.  Averaged across AM and PM peaks.
    """
    print("[3/4] Computing congestion scores...")

    edges["congestion_am"] = 1 - (edges["speed_am"] / edges["speed_freeflow"])
    edges["congestion_pm"] = 1 - (edges["speed_pm"] / edges["speed_freeflow"])
    edges["congestion_avg"] = (edges["congestion_am"] + edges["congestion_pm"]) / 2

    # Bin into severity labels
    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["Free flow", "Light", "Moderate", "Heavy", "Severe"]
    edges["severity"] = pd.cut(
        edges["congestion_avg"], bins=bins, labels=labels, include_lowest=True
    )

    # Save processed data
    summary = edges[["name", "highway", "speed_limit",
                      "speed_am", "speed_pm",
                      "congestion_am", "congestion_pm",
                      "congestion_avg", "severity"]].copy()
    summary.to_csv(DATA_DIR / "congestion_scores.csv", index=False)
    print(f"      Congestion scores saved to data/congestion_scores.csv")
    return edges


# ── Step 4: Build Folium map ──────────────────────────────────────────────────
def build_map(edges: pd.DataFrame):
    """Render an interactive Folium map colour-coded by congestion severity."""
    print("[4/4] Building interactive map...")

    COLOR_MAP = {
        "Free flow": "#22c55e",
        "Light":     "#facc15",
        "Moderate":  "#f97316",
        "Heavy":     "#ef4444",
        "Severe":    "#a855f7",
    }

    # Centre on Windhoek CBD
    m = folium.Map(location=[-22.5597, 17.0832], zoom_start=13, tiles="CartoDB dark_matter")

    # Add road segments
    for _, row in edges.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        coords = [(lat, lon) for lon, lat in geom.coords]
        severity = str(row.get("severity", "Free flow"))
        color = COLOR_MAP.get(severity, "#94a3b8")

        road_name = str(row.get("name", "Unnamed road"))
        popup_html = f"""
            <b>{road_name}</b><br>
            Type: {row.get('highway', 'N/A')}<br>
            Speed limit: {row.get('speed_limit', 'N/A')} km/h<br>
            AM peak speed: {row.get('speed_am', 0):.1f} km/h<br>
            PM peak speed: {row.get('speed_pm', 0):.1f} km/h<br>
            Congestion: <span style='color:{color}'><b>{severity}</b></span>
        """

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=3.5,
            opacity=0.8,
            tooltip=road_name,
            popup=folium.Popup(popup_html, max_width=220),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:#071e34; padding:12px 16px; border-radius:8px;
                border:1px solid #0e7490; font-family:Arial; font-size:13px;
                color:#e2f0f9;">
        <b style="color:#06b6d4;">Congestion Level</b><br>
        <span style='color:#22c55e'>&#9644;</span> Free flow<br>
        <span style='color:#facc15'>&#9644;</span> Light<br>
        <span style='color:#f97316'>&#9644;</span> Moderate<br>
        <span style='color:#ef4444'>&#9644;</span> Heavy<br>
        <span style='color:#a855f7'>&#9644;</span> Severe
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = OUTPUT_DIR / "windhoek_congestion_map.html"
    m.save(str(out_path))
    print(f"      Map saved to outputs/windhoek_congestion_map.html")
    return out_path


# ── Step 5: Summary stats ─────────────────────────────────────────────────────
def print_summary(edges: pd.DataFrame):
    print("\n── Congestion Summary ───────────────────────────────")
    counts = edges["severity"].value_counts().sort_index()
    for label, count in counts.items():
        pct = 100 * count / len(edges)
        print(f"  {label:<12} {count:>5} segments  ({pct:.1f}%)")
    most_congested = (
        edges.assign(highway=edges['highway'].astype(str))
        .groupby('highway')['congestion_avg'].mean()
        .idxmax()
    )
    print(f"\n  Most congested road type: {most_congested}")
    print("─────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    edges = fetch_road_network()
    edges = simulate_gps_observations(edges)
    edges = compute_congestion(edges)
    print_summary(edges)
    build_map(edges)
    print("Pipeline complete.")
