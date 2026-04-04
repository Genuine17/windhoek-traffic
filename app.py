"""
Windhoek Traffic Congestion Dashboard
======================================
Streamlit app that wraps the analysis pipeline with an interactive UI.
Theme: Sunset Orange & Purple — Professional / Corporate
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
    /* ── Global background & text ── */
    .stApp {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 40%, #3d1f3f 70%, #5c2d1e 100%);
        color: #f0e6d3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f0d35 0%, #2a1040 100%);
        border-right: 1px solid #ff6b35;
    }
    [data-testid="stSidebar"] * {
        color: #f0e6d3 !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #ff6b35 !important;
    }

    /* ── Header banner ── */
    .dashboard-header {
        background: linear-gradient(90deg, #c0392b 0%, #e84118 25%, #ff6b35 55%, #9b59b6 85%, #6c3483 100%);
        padding: 22px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.35);
    }
    .dashboard-header h1 {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .dashboard-header p {
        color: #ffe0cc !important;
        font-size: 0.92rem;
        margin: 6px 0 0 0;
        opacity: 0.9;
    }

    /* ── Section subheaders ── */
    h2, h3, .stSubheader {
        color: #ff9a6c !important;
        border-bottom: 2px solid #c0392b;
        padding-bottom: 4px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,107,53,0.12) 0%, rgba(155,89,182,0.12) 100%);
        border: 1px solid rgba(255, 107, 53, 0.35);
        border-left: 4px solid #ff6b35;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        transition: transform 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-left-color: #9b59b6;
        box-shadow: 0 4px 18px rgba(255, 107, 53, 0.25);
    }
    [data-testid="stMetricLabel"] {
        color: #ffb899 !important;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] {
        color: #ffb347 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #c0392b, #e84118, #ff6b35) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 3px 14px rgba(255, 107, 53, 0.45);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #9b59b6, #8e44ad) !important;
        box-shadow: 0 4px 18px rgba(155, 89, 182, 0.5);
        transform: translateY(-1px);
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255, 107, 53, 0.3) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(255, 107, 53, 0.07);
        border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 10px;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 107, 53, 0.3) !important;
        border-radius: 8px !important;
    }

    /* ── Info / success boxes ── */
    [data-testid="stAlert"] {
        background: rgba(155, 89, 182, 0.15) !important;
        border-left: 4px solid #9b59b6 !important;
        color: #f0e6d3 !important;
        border-radius: 8px;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #ff6b35 !important;
    }

    /* ── Severity label classes ── */
    .severity-free   { color: #2ecc71; font-weight: 600; }
    .severity-light  { color: #f39c12; font-weight: 600; }
    .severity-mod    { color: #ff6b35; font-weight: 600; }
    .severity-heavy  { color: #e74c3c; font-weight: 600; }
    .severity-severe { color: #9b59b6; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def flatten_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any list-valued cells to comma-separated strings.
    OSMnx often stores road names and highway types as lists when multiple
    values apply to a single segment; PyArrow cannot serialise raw lists."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
            )
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>🗺️ Windhoek Traffic Congestion Analysis</h1>
    <p>Peak-hour congestion modelling using OpenStreetMap road network data</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 🕐 Peak Hours")
    am_start, am_end = st.slider("AM Peak window", 5, 12, (7, 9),
                                  format="%d:00")
    pm_start, pm_end = st.slider("PM Peak window", 12, 22, (16, 18),
                                  format="%d:00")

    st.markdown("### 🚦 Congestion Model")
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
    st.success(f"✅ Analysis complete — {len(edges):,} road segments processed.")

# ── Results ───────────────────────────────────────────────────────────────────
edges = st.session_state.edges

if edges is not None:

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.subheader("📊 Summary")
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
        st.subheader("🗺️ Congestion Map")

        COLOR_MAP = {
            "Free flow": "#2ecc71",
            "Light":     "#f39c12",
            "Moderate":  "#ff6b35",
            "Heavy":     "#e74c3c",
            "Severe":    "#9b59b6",
        }

        m = folium.Map(location=[-22.5597, 17.0832], zoom_start=13,
                       tiles="CartoDB dark_matter")

        for _, row in edges.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            coords   = [(lat, lon) for lon, lat in geom.coords]
            severity = str(row.get("severity", "Free flow"))
            color    = COLOR_MAP.get(severity, "#95a5a6")
            name_val = row.get("name", "Unnamed road")
            name     = ", ".join(map(str, name_val)) if isinstance(name_val, list) else str(name_val)

            popup_html = f"""
                <div style="font-family:Arial;font-size:13px;min-width:180px;">
                <b style="color:#ff6b35;font-size:14px;">{name}</b><br>
                <hr style="border-color:#ddd;margin:4px 0;">
                <b>Type:</b> {row.get('highway','N/A')}<br>
                <b>Speed limit:</b> {row.get('speed_limit','N/A')} km/h<br>
                <b>AM speed:</b> {row.get('speed_am',0):.1f} km/h<br>
                <b>PM speed:</b> {row.get('speed_pm',0):.1f} km/h<br>
                <b>Severity:</b> <span style='color:{color};font-weight:700'>{severity}</span>
                </div>
            """
            folium.PolyLine(
                locations=coords, color=color, weight=4, opacity=0.85,
                tooltip=folium.Tooltip(name, style="font-family:Arial;font-size:13px;"),
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:linear-gradient(135deg,#1a0a2e,#2d1b4e);
                    padding:14px 18px;border-radius:10px;
                    border:1px solid #ff6b35;font-family:Arial;font-size:13px;
                    color:#f0e6d3;box-shadow:0 4px 16px rgba(0,0,0,0.5);">
            <b style="color:#ff9a6c;font-size:14px;">Congestion Level</b><br><br>
            <span style='color:#2ecc71;font-size:16px;'>&#9644;</span>&nbsp; Free flow<br>
            <span style='color:#f39c12;font-size:16px;'>&#9644;</span>&nbsp; Light<br>
            <span style='color:#ff6b35;font-size:16px;'>&#9644;</span>&nbsp; Moderate<br>
            <span style='color:#e74c3c;font-size:16px;'>&#9644;</span>&nbsp; Heavy<br>
            <span style='color:#9b59b6;font-size:16px;'>&#9644;</span>&nbsp; Severe
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, height=520, width="100%")

    with chart_col:
        st.subheader("🛣️ Breakdown by Road Type")

        road_congestion = (
            edges.assign(highway=edges["highway"].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
            ))
            .groupby("highway")["congestion_avg"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        road_congestion.columns = ["Road type", "Avg congestion"]
        road_congestion["Avg congestion"] = road_congestion["Avg congestion"].round(3)

        st.bar_chart(road_congestion.set_index("Road type"), height=240,
                     color="#ff6b35")

        st.subheader("🌅 AM vs PM Peak")
        peak_df = pd.DataFrame({
            "AM Peak": [edges["congestion_am"].mean()],
            "PM Peak": [edges["congestion_pm"].mean()],
        })
        st.bar_chart(peak_df, height=160, color=["#ff6b35", "#9b59b6"])

        st.subheader("📈 Severity Distribution")
        severity_order = ["Free flow", "Light", "Moderate", "Heavy", "Severe"]
        sev_df = (
            edges["severity"]
            .value_counts()
            .reindex(severity_order)
            .fillna(0)
            .reset_index()
        )
        sev_df.columns = ["Severity", "Count"]
        st.bar_chart(sev_df.set_index("Severity"), height=200,
                     color="#9b59b6")

    st.divider()

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 View raw congestion data"):
        display_cols = ["name", "highway", "speed_limit",
                        "speed_am", "speed_pm", "congestion_avg", "severity"]
        df_display = edges[display_cols].copy()
        df_display["speed_am"]       = df_display["speed_am"].round(1)
        df_display["speed_pm"]       = df_display["speed_pm"].round(1)
        df_display["congestion_avg"] = df_display["congestion_avg"].round(3)
        df_display = flatten_list_columns(df_display)

        st.dataframe(df_display, width="stretch", height=300)

        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv,
                           "windhoek_congestion.csv", "text/csv",
                           type="primary")

else:
    # ── Welcome landing screen ─────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(155,89,182,0.12));
        border: 1px solid rgba(255,107,53,0.35);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
        text-align: center;
    ">
        <h2 style="color:#ff9a6c;margin-bottom:6px;">Welcome to the Windhoek Traffic Dashboard</h2>
        <p style="color:#f0e6d3;font-size:1.05rem;opacity:0.85;">
            Use the sidebar to configure peak hours and click <b style="color:#ff6b35;">▶ Run Analysis</b> to generate a live congestion map.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key facts ─────────────────────────────────────────────────────────────
    st.subheader("📌 Key Facts About Windhoek Traffic")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,107,53,0.13),rgba(155,89,182,0.1));
                    border:1px solid rgba(255,107,53,0.3);border-radius:12px;padding:20px 22px;">
            <div style="font-size:2rem;">🏙️</div>
            <div style="color:#ff9a6c;font-weight:700;font-size:1rem;margin:8px 0 4px;">Population</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;">~450,000</div>
            <div style="color:#f0e6d3;font-size:0.85rem;opacity:0.75;">Windhoek is home to nearly half a million residents, making it the largest city in Namibia.</div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,107,53,0.13),rgba(155,89,182,0.1));
                    border:1px solid rgba(255,107,53,0.3);border-radius:12px;padding:20px 22px;">
            <div style="font-size:2rem;">🚗</div>
            <div style="color:#ff9a6c;font-weight:700;font-size:1rem;margin:8px 0 4px;">Registered Vehicles</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;">200,000+</div>
            <div style="color:#f0e6d3;font-size:0.85rem;opacity:0.75;">Over 200,000 vehicles are registered in Windhoek, putting pressure on the city's road network daily.</div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,107,53,0.13),rgba(155,89,182,0.1));
                    border:1px solid rgba(255,107,53,0.3);border-radius:12px;padding:20px 22px;">
            <div style="font-size:2rem;">⏰</div>
            <div style="color:#ff9a6c;font-weight:700;font-size:1rem;margin:8px 0 4px;">Peak Hours</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;">07:00 – 09:00</div>
            <div style="color:#f0e6d3;font-size:0.85rem;opacity:0.75;">Morning and afternoon peaks (16:00–18:00) see the heaviest congestion on arterial roads.</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    f4, f5, f6 = st.columns(3)
    with f4:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,107,53,0.13),rgba(155,89,182,0.1));
                    border:1px solid rgba(255,107,53,0.3);border-radius:12px;padding:20px 22px;">
            <div style="font-size:2rem;">🛣️</div>
            <div style="color:#ff9a6c;font-weight:700;font-size:1rem;margin:8px 0 4px;">Road Network</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;">1,000+ km</div>
            <div style="color:#f0e6d3;font-size:0.85rem;opacity:0.75;">Windhoek's drivable road network spans over 1,000 km of streets, from highways to residential roads.</div>
        </div>
        """, unsafe_allow_html=True)

    with f5:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,107,53,0.13),rgba(155,89,182,0.1));
                    border:1px solid rgba(255,107,53,0.3);border-radius:12px;padding:20px 22px;">
            <div style="font-size:2rem;">📈</div>
            <div style="color:#ff9a6c;font-weight:700;font-size:1rem;margin:8px 0 4px;">Traffic Growth</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;">~5% / year</div>
            <div style="color:#f0e6d3;font-size:0.85rem;opacity:0.75;">Vehicle numbers in Windhoek grow by approximately 5% annually, steadily increasing congestion.</div>
        </div>
        """, unsafe_allow_html=True)

    with f6:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,107,53,0.13),rgba(155,89,182,0.1));
                    border:1px solid rgba(255,107,53,0.3);border-radius:12px;padding:20px 22px;">
            <div style="font-size:2rem;">🗺️</div>
            <div style="color:#ff9a6c;font-weight:700;font-size:1rem;margin:8px 0 4px;">Data Source</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;">OpenStreetMap</div>
            <div style="color:#f0e6d3;font-size:0.85rem;opacity:0.75;">This dashboard uses live road network data from OpenStreetMap, updated by contributors worldwide.</div>
        </div>
        """, unsafe_allow_html=True)
