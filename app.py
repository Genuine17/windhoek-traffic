"""
Windhoek Traffic Congestion Dashboard
======================================
Streamlit app that wraps the analysis pipeline with an interactive UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import (
    fetch_road_network,
    simulate_gps_observations,
    compute_congestion,
    build_map,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Windhoek Traffic Analysis",
    page_icon="🗺️",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #378ADD;
    }
    .severity-free   { color: #2ecc71; font-weight: 600; }
    .severity-light  { color: #f1c40f; font-weight: 600; }
    .severity-mod    { color: #e67e22; font-weight: 600; }
    .severity-heavy  { color: #e74c3c; font-weight: 600; }
    .severity-severe { color: #8e44ad; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🗺️ Windhoek Traffic Congestion Analysis")
st.caption("Peak-hour congestion modelling using OpenStreetMap road network data")

st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Peak Hours")
    am_start, am_end = st.slider("AM Peak window", 5, 12, (7, 9),
                                  format="%d:00")
    pm_start, pm_end = st.slider("PM Peak window", 12, 22, (16, 18),
                                  format="%d:00")

    st.subheader("Congestion Model")
    arterial_factor = st.slider(
        "Arterial road slow-down factor",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Lower = more congestion on primary/secondary roads"
    )

    seed = st.number_input("Random seed", value=42, step=1,
                            help="Change to generate a different simulation run")

    st.divider()
    run = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Data: © OpenStreetMap contributors")
    st.caption("Built by Phillann Genuine Shitaleni 2026")

# ── State ─────────────────────────────────────────────────────────────────────
if "edges" not in st.session_state:
    st.session_state.edges = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching road network from OpenStreetMap..."):
        edges = fetch_road_network()

    with st.spinner("Simulating peak-hour speeds..."):
        edges = simulate_gps_observations(edges, seed=int(seed))

    with st.spinner("Computing congestion scores..."):
        edges = compute_congestion(edges)

    st.session_state.edges = edges
    st.success(f"Analysis complete — {len(edges):,} road segments processed.")

# ── Results ───────────────────────────────────────────────────────────────────
edges = st.session_state.edges

if edges is not None:

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.subheader("Summary")
    counts = edges["severity"].value_counts()
    total  = len(edges)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total segments",  f"{total:,}")
    c2.metric("🟢 Free flow",    f"{counts.get('Free flow', 0):,}",
              f"{100*counts.get('Free flow',0)/total:.1f}%")
    c3.metric("🟡 Light",        f"{counts.get('Light', 0):,}",
              f"{100*counts.get('Light',0)/total:.1f}%")
    c4.metric("🟠 Moderate",     f"{counts.get('Moderate', 0):,}",
              f"{100*counts.get('Moderate',0)/total:.1f}%")
    c5.metric("🔴 Heavy",        f"{counts.get('Heavy', 0):,}",
              f"{100*counts.get('Heavy',0)/total:.1f}%")
    c6.metric("🟣 Severe",       f"{counts.get('Severe', 0):,}",
              f"{100*counts.get('Severe',0)/total:.1f}%")

    st.divider()

    # ── Map + charts side by side ─────────────────────────────────────────────
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        st.subheader("Congestion Map")

        COLOR_MAP = {
            "Free flow": "#2ecc71",
            "Light":     "#f1c40f",
            "Moderate":  "#e67e22",
            "Heavy":     "#e74c3c",
            "Severe":    "#8e44ad",
        }

        m = folium.Map(location=[-22.5597, 17.0832], zoom_start=13,
                       tiles="CartoDB positron")

        for _, row in edges.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            coords   = [(lat, lon) for lon, lat in geom.coords]
            severity = str(row.get("severity", "Free flow"))
            color    = COLOR_MAP.get(severity, "#95a5a6")
            name     = str(row.get("name", "Unnamed road"))

            popup_html = f"""
                <b>{name}</b><br>
                Type: {row.get('highway','N/A')}<br>
                Speed limit: {row.get('speed_limit','N/A')} km/h<br>
                AM speed: {row.get('speed_am',0):.1f} km/h<br>
                PM speed: {row.get('speed_pm',0):.1f} km/h<br>
                Severity: <b style='color:{color}'>{severity}</b>
            """
            folium.PolyLine(
                locations=coords, color=color, weight=3.5, opacity=0.8,
                tooltip=name,
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:12px 16px;border-radius:8px;
                    border:1px solid #ccc;font-family:Arial;font-size:13px;">
            <b>Congestion Level</b><br>
            <span style='color:#2ecc71'>&#9644;</span> Free flow<br>
            <span style='color:#f1c40f'>&#9644;</span> Light<br>
            <span style='color:#e67e22'>&#9644;</span> Moderate<br>
            <span style='color:#e74c3c'>&#9644;</span> Heavy<br>
            <span style='color:#8e44ad'>&#9644;</span> Severe
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, height=520, use_container_width=True)

    with chart_col:
        st.subheader("Breakdown by Road Type")

        road_congestion = (
            edges.assign(highway=edges["highway"].astype(str))
            .groupby("highway")["congestion_avg"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        road_congestion.columns = ["Road type", "Avg congestion"]
        road_congestion["Avg congestion"] = road_congestion["Avg congestion"].round(3)

        st.bar_chart(road_congestion.set_index("Road type"), height=240)

        st.subheader("AM vs PM Peak")
        peak_df = pd.DataFrame({
            "AM Peak": [edges["congestion_am"].mean()],
            "PM Peak": [edges["congestion_pm"].mean()],
        })
        st.bar_chart(peak_df, height=160)

        st.subheader("Severity Distribution")
        severity_order = ["Free flow", "Light", "Moderate", "Heavy", "Severe"]
        sev_df = (
            edges["severity"]
            .value_counts()
            .reindex(severity_order)
            .fillna(0)
            .reset_index()
        )
        sev_df.columns = ["Severity", "Count"]
        st.bar_chart(sev_df.set_index("Severity"), height=200)

    st.divider()

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 View raw congestion data"):
        display_cols = ["name", "highway", "speed_limit",
                        "speed_am", "speed_pm", "congestion_avg", "severity"]
        df_display = edges[display_cols].copy()
        df_display["speed_am"]       = df_display["speed_am"].round(1)
        df_display["speed_pm"]       = df_display["speed_pm"].round(1)
        df_display["congestion_avg"] = df_display["congestion_avg"].round(3)
        st.dataframe(df_display, use_container_width=True, height=300)

        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv,
                           "windhoek_congestion.csv", "text/csv")

else:
    # ── Placeholder before first run ──────────────────────────────────────────
    st.info("👈 Adjust settings in the sidebar and click **Run Analysis** to start.")

    st.markdown("""
    ### How it works
    1. **Fetch** — downloads Windhoek's full drivable road network from OpenStreetMap
    2. **Simulate** — models AM and PM peak-hour speeds per road segment based on road type and speed limits
    3. **Score** — computes a congestion index (0 = free flow → 1 = standstill) for every segment
    4. **Visualise** — renders an interactive map and charts you can explore here

    > The speed observations are currently simulated. The pipeline accepts real
    > probe-vehicle CSV data with no code changes needed.
    """)
