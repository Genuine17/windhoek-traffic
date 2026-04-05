"""
Windhoek Traffic Congestion Dashboard
======================================
Streamlit app that wraps the analysis pipeline with an interactive UI.
Theme: Deep Blue & Teal — Professional / Corporate
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

# ── Design tokens ─────────────────────────────────────────────────────────────
# Primary:  #0a2540  (deep navy)
# Accent:   #0e7490  (teal)
# Accent 2: #06b6d4  (cyan highlight)
# Surface:  #0f2d48  (card/panel)
# Text:     #e2f0f9  (light blue-white)

st.markdown("""
<style>
  /* ── Viewport meta for mobile ── */
  @viewport { width=device-width; initial-scale=1; }

  /* ── Global ── */
  .stApp {
      background-color: #06172b;
      background-image:
          radial-gradient(ellipse at 20% 20%, rgba(6,182,212,0.07) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 80%, rgba(14,116,144,0.06) 0%, transparent 50%);
      color: #e2f0f9;
      font-family: 'Segoe UI', system-ui, sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background-color: #071e34;
      border-right: 2px solid #0e7490;
  }
  [data-testid="stSidebar"] * { color: #cce8f4 !important; }
  [data-testid="stSidebar"] .stSlider > div > div > div {
      background: #06b6d4 !important;
  }
  [data-testid="stSidebar"] input[type="number"],
  [data-testid="stSidebar"] .stNumberInput input {
      background-color: #0a2540 !important;
      color: #e2f0f9 !important;
      border: 1px solid #0e7490 !important;
      border-radius: 6px !important;
  }
  [data-testid="stSidebar"] .stNumberInput button {
      background-color: #0a2540 !important;
      color: #06b6d4 !important;
      border: 1px solid #0e7490 !important;
  }
  [data-testid="stSidebar"] .stNumberInput button:hover {
      background-color: #0e7490 !important;
      color: #ffffff !important;
  }

  /* ── Header banner ── */
  .dashboard-header {
      background: linear-gradient(90deg, #0a2540 0%, #0e7490 50%, #06b6d4 100%);
      padding: 24px 36px;
      border-radius: 14px;
      margin-bottom: 28px;
      box-shadow: 0 4px 28px rgba(6,182,212,0.25);
      border: 1px solid rgba(6,182,212,0.2);
  }
  .dashboard-header h1 {
      color: #ffffff !important;
      font-size: 2rem;
      font-weight: 700;
      margin: 0;
      letter-spacing: 0.4px;
  }
  .dashboard-header p {
      color: #b0dff0 !important;
      font-size: 0.93rem;
      margin: 7px 0 0 0;
  }

  /* ── Subheaders ── */
  h2, h3 {
      color: #06b6d4 !important;
      border-bottom: 2px solid rgba(6,182,212,0.25);
      padding-bottom: 5px;
  }

  /* ── Metric cards — white with teal accent, smooth fill on hover ── */
  [data-testid="stMetric"] {
      background: #ffffff;
      border: none;
      border-left: 4px solid #0e7490;
      border-radius: 10px;
      padding: 16px 18px;
      box-shadow: 0 2px 14px rgba(0,0,0,0.35);
      transition: background 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
      cursor: default;
  }
  [data-testid="stMetric"]:hover {
      background: #0e7490;
      box-shadow: 0 6px 24px rgba(14,116,144,0.45);
  }
  [data-testid="stMetric"]:hover [data-testid="stMetricLabel"],
  [data-testid="stMetric"]:hover [data-testid="stMetricValue"],
  [data-testid="stMetric"]:hover [data-testid="stMetricDelta"] {
      color: #ffffff !important;
  }
  [data-testid="stMetricLabel"] {
      color: #4a6580 !important;
      font-weight: 700;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      transition: color 0.3s ease;
  }
  [data-testid="stMetricValue"] {
      color: #071e34 !important;
      font-weight: 800;
      transition: color 0.3s ease;
  }
  [data-testid="stMetricDelta"] {
      color: #0e7490 !important;
      font-weight: 600;
      transition: color 0.3s ease;
  }

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {
      background: linear-gradient(90deg, #0a2540, #0e7490) !important;
      color: #ffffff !important;
      border: 1px solid #06b6d4 !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      letter-spacing: 0.5px;
      box-shadow: 0 3px 14px rgba(6,182,212,0.3);
      transition: all 0.25s ease;
  }
  .stButton > button[kind="primary"]:hover {
      background: linear-gradient(90deg, #0e7490, #06b6d4) !important;
      box-shadow: 0 5px 20px rgba(6,182,212,0.5);
      transform: translateY(-1px);
  }

  /* ── Divider ── */
  hr { border-color: rgba(6,182,212,0.2) !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] {
      background: rgba(6,182,212,0.05);
      border: 1px solid rgba(6,182,212,0.2);
      border-radius: 10px;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
      border: 1px solid rgba(6,182,212,0.25) !important;
      border-radius: 8px !important;
  }

  /* ── Alert / info / success ── */
  [data-testid="stAlert"] {
      background: rgba(14,116,144,0.15) !important;
      border-left: 4px solid #06b6d4 !important;
      color: #e2f0f9 !important;
      border-radius: 8px;
  }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #06b6d4 !important; }

  /* ── Responsive flexbox fact cards ── */
  .fact-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 8px;
  }
  .fact-card {
      flex: 1 1 260px;
      background: #0f2d48;
      border: 1px solid rgba(6,182,212,0.2);
      border-top: 3px solid #0e7490;
      border-radius: 12px;
      padding: 20px 22px;
      box-sizing: border-box;
      transition: background 0.3s ease, border-top-color 0.3s ease, box-shadow 0.3s ease;
  }
  .fact-card:hover {
      background: #0e7490;
      border-top-color: #06b6d4;
      box-shadow: 0 8px 28px rgba(6,182,212,0.3);
  }
  .fact-card:hover .fact-label { color: #b0f0ff !important; }
  .fact-card:hover .fact-value { color: #ffffff !important; }
  .fact-card:hover .fact-desc  { color: #d0f4ff !important; opacity: 1; }
  .fact-icon  { font-size: 2rem; margin-bottom: 10px; }
  .fact-label { color: #06b6d4; font-weight: 700; font-size: 0.88rem;
                text-transform: uppercase; letter-spacing: 0.6px;
                margin-bottom: 4px; transition: color 0.3s ease; }
  .fact-value { color: #ffffff; font-size: 1.65rem; font-weight: 800;
                margin-bottom: 6px; transition: color 0.3s ease; }
  .fact-desc  { color: #8ab8d4; font-size: 0.84rem; line-height: 1.5;
                opacity: 0.85; transition: color 0.3s ease, opacity 0.3s ease; }

  /* ── Welcome banner ── */
  .welcome-banner {
      background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(14,116,144,0.1));
      border: 1px solid rgba(6,182,212,0.25);
      border-radius: 14px;
      padding: 28px 36px;
      margin-bottom: 28px;
      text-align: center;
  }
  .welcome-banner h2 { color: #06b6d4 !important; border: none !important; }
  .welcome-banner p  { color: #b0dff0; font-size: 1.05rem; margin-top: 8px; }

  /* ── Mobile breakpoint ── */
  @media (max-width: 640px) {
      .dashboard-header h1 { font-size: 1.4rem; }
      .fact-card { flex: 1 1 100%; }
      [data-testid="stMetric"] { padding: 12px 14px; }
  }

  /* ── Severity label classes ── */
  .severity-free   { color: #22c55e; font-weight: 600; }
  .severity-light  { color: #facc15; font-weight: 600; }
  .severity-mod    { color: #f97316; font-weight: 600; }
  .severity-heavy  { color: #ef4444; font-weight: 600; }
  .severity-severe { color: #a855f7; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def flatten_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any list-valued cells to comma-separated strings."""
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### 🕐 Peak Hours")
    am_start, am_end = st.slider("AM Peak window", 5, 12, (7, 9), format="%d:00")
    pm_start, pm_end = st.slider("PM Peak window", 12, 22, (16, 18), format="%d:00")

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

# ── Pipeline ──────────────────────────────────────────────────────────────────
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
    c1.metric("Total segments", f"{total:,}")
    c2.metric("🟢 Free flow",   f"{counts.get('Free flow', 0):,}",
              f"{100*counts.get('Free flow',0)/total:.1f}%")
    c3.metric("🟡 Light",       f"{counts.get('Light', 0):,}",
              f"{100*counts.get('Light',0)/total:.1f}%")
    c4.metric("🟠 Moderate",    f"{counts.get('Moderate', 0):,}",
              f"{100*counts.get('Moderate',0)/total:.1f}%")
    c5.metric("🔴 Heavy",       f"{counts.get('Heavy', 0):,}",
              f"{100*counts.get('Heavy',0)/total:.1f}%")
    c6.metric("🟣 Severe",      f"{counts.get('Severe', 0):,}",
              f"{100*counts.get('Severe',0)/total:.1f}%")

    st.divider()

    # ── Map + charts ──────────────────────────────────────────────────────────
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        st.subheader("🗺️ Congestion Map")

        COLOR_MAP = {
            "Free flow": "#22c55e",
            "Light":     "#facc15",
            "Moderate":  "#f97316",
            "Heavy":     "#ef4444",
            "Severe":    "#a855f7",
        }

        m = folium.Map(location=[-22.5597, 17.0832], zoom_start=13,
                       tiles="CartoDB dark_matter")

        for _, row in edges.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            coords   = [(lat, lon) for lon, lat in geom.coords]
            severity = str(row.get("severity", "Free flow"))
            color    = COLOR_MAP.get(severity, "#94a3b8")
            name_val = row.get("name", "Unnamed road")
            name     = ", ".join(map(str, name_val)) if isinstance(name_val, list) else str(name_val)

            popup_html = f"""
                <div style="font-family:'Segoe UI',Arial;font-size:13px;min-width:190px;
                            background:#0f2d48;color:#e2f0f9;padding:12px 14px;border-radius:8px;">
                <b style="color:#06b6d4;font-size:14px;">{name}</b><br>
                <hr style="border-color:rgba(6,182,212,0.3);margin:6px 0;">
                <b>Type:</b> {row.get('highway','N/A')}<br>
                <b>Speed limit:</b> {row.get('speed_limit','N/A')} km/h<br>
                <b>AM speed:</b> {row.get('speed_am',0):.1f} km/h<br>
                <b>PM speed:</b> {row.get('speed_pm',0):.1f} km/h<br>
                <b>Severity:</b> <span style='color:{color};font-weight:700'>{severity}</span>
                </div>
            """
            folium.PolyLine(
                locations=coords, color=color, weight=4, opacity=0.85,
                tooltip=folium.Tooltip(name, style="font-family:Segoe UI;font-size:13px;"),
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#071e34;padding:14px 18px;border-radius:10px;
                    border:1px solid #0e7490;font-family:'Segoe UI',Arial;font-size:13px;
                    color:#e2f0f9;box-shadow:0 4px 18px rgba(0,0,0,0.6);">
            <b style="color:#06b6d4;font-size:14px;">Congestion Level</b><br><br>
            <span style='color:#22c55e;font-size:16px;'>&#9644;</span>&nbsp; Free flow<br>
            <span style='color:#facc15;font-size:16px;'>&#9644;</span>&nbsp; Light<br>
            <span style='color:#f97316;font-size:16px;'>&#9644;</span>&nbsp; Moderate<br>
            <span style='color:#ef4444;font-size:16px;'>&#9644;</span>&nbsp; Heavy<br>
            <span style='color:#a855f7;font-size:16px;'>&#9644;</span>&nbsp; Severe
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
        st.bar_chart(road_congestion.set_index("Road type"), height=240, color="#0e7490")

        st.subheader("🌊 AM vs PM Peak")
        peak_df = pd.DataFrame({
            "AM Peak": [edges["congestion_am"].mean()],
            "PM Peak": [edges["congestion_pm"].mean()],
        })
        st.bar_chart(peak_df, height=160, color=["#0e7490", "#06b6d4"])

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
        st.bar_chart(sev_df.set_index("Severity"), height=200, color="#06b6d4")

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
                           "windhoek_congestion.csv", "text/csv", type="primary")

else:
    # ── Welcome landing screen ─────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-banner">
        <h2>Welcome to the Windhoek Traffic Dashboard</h2>
        <p>Use the sidebar to configure peak hours and click
           <b style="color:#06b6d4;">▶ Run Analysis</b>
           to generate a live congestion map.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📌 Key Facts About Windhoek Traffic")

    st.markdown("""
    <div class="fact-grid">
        <div class="fact-card">
            <div class="fact-icon">🏙️</div>
            <div class="fact-label">Population</div>
            <div class="fact-value">~450,000</div>
            <div class="fact-desc">Windhoek is home to nearly half a million residents,
            making it the largest city in Namibia.</div>
        </div>
        <div class="fact-card">
            <div class="fact-icon">🚗</div>
            <div class="fact-label">Registered Vehicles</div>
            <div class="fact-value">200,000+</div>
            <div class="fact-desc">Over 200,000 vehicles are registered in Windhoek,
            putting pressure on the city's road network daily.</div>
        </div>
        <div class="fact-card">
            <div class="fact-icon">⏰</div>
            <div class="fact-label">Peak Hours</div>
            <div class="fact-value">07:00 – 09:00</div>
            <div class="fact-desc">Morning and afternoon peaks (16:00–18:00) see
            the heaviest congestion on arterial roads.</div>
        </div>
    </div>
    <div class="fact-grid">
        <div class="fact-card">
            <div class="fact-icon">🛣️</div>
            <div class="fact-label">Road Network</div>
            <div class="fact-value">1,000+ km</div>
            <div class="fact-desc">Windhoek's drivable road network spans over 1,000 km
            of streets, from highways to residential roads.</div>
        </div>
        <div class="fact-card">
            <div class="fact-icon">📈</div>
            <div class="fact-label">Traffic Growth</div>
            <div class="fact-value">~5% / year</div>
            <div class="fact-desc">Vehicle numbers in Windhoek grow by approximately
            5% annually, steadily increasing congestion.</div>
        </div>
        <div class="fact-card">
            <div class="fact-icon">🗺️</div>
            <div class="fact-label">Data Source</div>
            <div class="fact-value">OpenStreetMap</div>
            <div class="fact-desc">This dashboard uses live road network data from
            OpenStreetMap, updated by contributors worldwide.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
