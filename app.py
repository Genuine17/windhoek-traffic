"""
Windhoek Traffic Congestion Dashboard
======================================
Theme: Warm Earth & Brown — inspired by Namibian landscape
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys
import hashlib
import requests

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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  .stApp, .stApp * {
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }
  .stApp {
      background-color: #140a04;
      background-image:
          radial-gradient(ellipse at 15% 10%, rgba(196,154,108,0.06) 0%, transparent 45%),
          radial-gradient(ellipse at 85% 90%, rgba(139,94,60,0.05) 0%, transparent 45%);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1e0e05 0%, #160904 100%) !important;
      border-right: 1px solid #5a3018 !important;
  }
  [data-testid="stSidebar"] h2 {
      color: #f0dfc8 !important; font-size: 1.05rem !important;
      font-weight: 700 !important; border-bottom: 1px solid #5a3018 !important;
      padding-bottom: 8px !important;
  }
  [data-testid="stSidebar"] h3 {
      color: #c49a6c !important; font-size: 0.8rem !important;
      font-weight: 600 !important; text-transform: uppercase;
      letter-spacing: 1px; border: none !important; margin-top: 16px !important;
  }

  /* Header */
  .dashboard-header {
      background: linear-gradient(135deg, #2c1508 0%, #5a3018 40%, #8b5e3c 75%, #c49a6c 100%);
      padding: 28px 40px; border-radius: 16px; margin-bottom: 28px;
      border: 1px solid rgba(196,154,108,0.25); position: relative; overflow: hidden;
  }
  .dashboard-header::before {
      content: ''; position: absolute; top: -40%; right: -5%;
      width: 280px; height: 280px;
      background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
      border-radius: 50%;
  }
  .dashboard-header h1 {
      color: #fff !important; font-size: 1.9rem !important;
      font-weight: 800 !important; margin: 0 !important;
      letter-spacing: -0.3px; border: none !important;
  }
  .dashboard-header p {
      color: #e8c99a !important; font-size: 0.9rem !important;
      margin: 8px 0 0 0 !important;
  }
  .header-badge {
      display: inline-block;
      background: rgba(196,154,108,0.18); border: 1px solid rgba(196,154,108,0.35);
      border-radius: 20px; padding: 3px 12px; font-size: 0.7rem;
      color: #f0dfc8; font-weight: 500; letter-spacing: 0.5px; margin-top: 10px;
  }

  /* Subheaders */
  h2, h3 {
      color: #c49a6c !important;
      border-bottom: 1px solid rgba(196,154,108,0.2) !important;
      padding-bottom: 6px !important; font-weight: 700 !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
      background: linear-gradient(145deg, #2a1508, #231005);
      border: 1px solid #5a3018; border-top: 3px solid #8b5e3c;
      border-radius: 12px; padding: 18px 20px;
      transition: all 0.25s ease; cursor: default;
  }
  [data-testid="stMetric"]:hover {
      border-top-color: #c49a6c; border-color: #8b5e3c;
      transform: translateY(-2px); box-shadow: 0 8px 24px rgba(139,94,60,0.3);
  }
  [data-testid="stMetricLabel"] {
      color: #8b7060 !important; font-weight: 600 !important;
      font-size: 0.7rem !important; text-transform: uppercase !important;
      letter-spacing: 0.8px !important; white-space: normal !important;
  }
  [data-testid="stMetricValue"] {
      color: #f0dfc8 !important; font-weight: 800 !important;
      font-size: clamp(1.1rem, 2vw, 1.5rem) !important;
  }
  [data-testid="stMetricDelta"] {
      color: #c49a6c !important; font-weight: 500 !important; font-size: 0.75rem !important;
  }

  /* Buttons */
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #8b5e3c, #c49a6c) !important;
      color: #1a0a03 !important; border: none !important; border-radius: 10px !important;
      font-weight: 700 !important; font-size: 0.88rem !important;
      padding: 10px 20px !important; transition: all 0.2s ease;
  }
  .stButton > button[kind="primary"]:hover {
      background: linear-gradient(135deg, #c49a6c, #e8b87a) !important;
      transform: translateY(-1px); box-shadow: 0 6px 20px rgba(196,154,108,0.4) !important;
  }
  .stButton > button[kind="secondary"],
  .stButton > button:not([kind]) {
      background: transparent !important; color: #c49a6c !important;
      border: 1px solid #5a3018 !important; border-radius: 8px !important;
      font-size: 0.82rem !important; transition: all 0.2s ease;
  }
  .stButton > button[kind="secondary"]:hover,
  .stButton > button:not([kind]):hover {
      background: rgba(139,94,60,0.15) !important;
      border-color: #8b5e3c !important; color: #e8b87a !important;
  }

  /* Text input */
  .stTextInput input {
      background: #1e0e05 !important; border: 1px solid #5a3018 !important;
      border-radius: 10px !important; color: #f0dfc8 !important;
      font-size: 0.9rem !important; padding: 10px 14px !important;
  }
  .stTextInput input:focus {
      border-color: #c49a6c !important;
      box-shadow: 0 0 0 2px rgba(196,154,108,0.15) !important;
  }
  .stTextInput input::placeholder { color: #6b4226 !important; }

  /* Divider */
  hr { border-color: rgba(196,154,108,0.15) !important; }

  /* Expander */
  [data-testid="stExpander"] {
      background: #1e0e05; border: 1px solid #3d1f0a !important;
      border-radius: 12px !important;
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #c49a6c !important; }

  /* Search panel */
  .search-panel {
      background: linear-gradient(145deg, #1e0e05, #2a1508);
      border: 1px solid #5a3018; border-radius: 14px;
      padding: 16px 18px; margin-bottom: 12px;
  }
  .search-panel-title {
      color: #c49a6c; font-weight: 700; font-size: 0.82rem;
      text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;
  }
  .quick-label {
      color: #7a5535; font-size: 0.72rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.6px; margin: 10px 0 6px;
  }

  /* Fact cards */
  .fact-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; }
  .fact-card {
      flex: 1 1 260px;
      background: linear-gradient(145deg, #2a1508, #1e0e05);
      border: 1px solid #3d1f0a; border-top: 3px solid #6b3d1e;
      border-radius: 14px; padding: 22px 24px; box-sizing: border-box;
      transition: all 0.28s ease;
  }
  .fact-card:hover {
      border-top-color: #c49a6c; border-color: #8b5e3c;
      transform: translateY(-3px); box-shadow: 0 10px 32px rgba(139,94,60,0.25);
  }
  .fact-icon  { font-size: 1.8rem; margin-bottom: 10px; }
  .fact-label { color: #8b6040; font-weight: 700; font-size: 0.76rem;
                text-transform: uppercase; letter-spacing: 0.8px;
                margin-bottom: 6px; transition: color 0.28s; }
  .fact-value { color: #f0dfc8; font-size: 1.6rem; font-weight: 800;
                margin-bottom: 8px; letter-spacing: -0.5px; transition: color 0.28s; }
  .fact-desc  { color: #7a5535; font-size: 0.82rem; line-height: 1.55;
                transition: color 0.28s; }
  .fact-card:hover .fact-label { color: #c49a6c; }
  .fact-card:hover .fact-value { color: #fff; }
  .fact-card:hover .fact-desc  { color: #e8c99a; }

  /* Welcome banner */
  .welcome-banner {
      background: linear-gradient(135deg, rgba(139,94,60,0.12), rgba(90,48,24,0.1));
      border: 1px solid rgba(196,154,108,0.2); border-left: 4px solid #8b5e3c;
      border-radius: 14px; padding: 30px 40px; margin-bottom: 28px;
  }
  .welcome-banner h2 {
      color: #f0dfc8 !important; border: none !important;
      font-size: 1.5rem !important; margin-bottom: 8px !important;
  }
  .welcome-banner p { color: #b89070; font-size: 1rem; margin: 0; }

  @media (max-width: 640px) {
      .dashboard-header { padding: 20px 22px; }
      .dashboard-header h1 { font-size: 1.35rem !important; }
      .fact-card { flex: 1 1 100%; }
  }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def flatten_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
            )
    return df


def _hash_geodataframe(gdf):
    h = hashlib.md5()
    h.update(str(gdf.shape).encode())
    try:
        h.update(gdf.geometry.to_wkt().str.cat().encode())
    except Exception:
        h.update(str(gdf.index.tolist()).encode())
    return h.hexdigest()


@st.cache_data(
    hash_funcs={"geopandas.geodataframe.GeoDataFrame": _hash_geodataframe}
)
def cached_simulate(edges, seed: int):
    return simulate_gps_observations(edges, seed=seed)


@st.cache_data(ttl=3600)
def geocode_location(query: str):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{query}, Windhoek, Namibia", "format": "json",
                    "limit": 5, "addressdetails": 1},
            headers={"User-Agent": "WindhoekTrafficDashboard/1.0"},
            timeout=6,
        )
        return [
            {"name": res.get("display_name", "").split(",")[0],
             "lat": float(res["lat"]), "lon": float(res["lon"])}
            for res in r.json()
        ]
    except Exception:
        return []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>🗺️ Windhoek Traffic Congestion Analysis</h1>
    <p>Peak-hour congestion modelling powered by OpenStreetMap road network data</p>
    <span class="header-badge">🇳🇦 Namibia &nbsp;·&nbsp; Live OSM Data &nbsp;·&nbsp; 2026</span>
</div>
""", unsafe_allow_html=True)

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
        help="Lower = more congestion on primary/secondary roads",
    )
    seed = st.number_input("Random seed", value=42, step=1,
                           help="Change to generate a different simulation run")

    st.divider()
    run = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("Data: © OpenStreetMap contributors")
    st.caption("Built by Phillann Genuine Shitaleni 2026")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("edges", None), ("map_center", [-22.5597, 17.0832]),
    ("map_zoom", 13), ("search_query", ""), ("search_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching road network from OpenStreetMap..."):
        edges = fetch_road_network()
    with st.spinner("Simulating peak-hour speeds..."):
        edges = cached_simulate(edges, seed=int(seed))
    with st.spinner("Computing congestion scores..."):
        edges = compute_congestion(edges)
    st.session_state.edges = edges
    st.success(f"✅ Analysis complete — {len(edges):,} road segments processed.")

# ── Results ───────────────────────────────────────────────────────────────────
edges = st.session_state.edges

if edges is not None:

    st.subheader("📊 Network Summary")
    counts = edges["severity"].value_counts()
    total  = len(edges)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Segments", f"{total:,}")
    c2.metric("🟢 Free flow",  f"{counts.get('Free flow',0):,}", f"{100*counts.get('Free flow',0)/total:.1f}%")
    c3.metric("🟡 Light",      f"{counts.get('Light',0):,}",     f"{100*counts.get('Light',0)/total:.1f}%")
    c4.metric("🟠 Moderate",   f"{counts.get('Moderate',0):,}",  f"{100*counts.get('Moderate',0)/total:.1f}%")
    c5.metric("🔴 Heavy",      f"{counts.get('Heavy',0):,}",     f"{100*counts.get('Heavy',0)/total:.1f}%")
    c6.metric("🟣 Severe",     f"{counts.get('Severe',0):,}",    f"{100*counts.get('Severe',0)/total:.1f}%")

    st.divider()
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        st.subheader("🗺️ Congestion Map")

        # Search panel
        st.markdown('<div class="search-panel"><div class="search-panel-title">🔍 Search Area</div></div>',
                    unsafe_allow_html=True)
        s1, s2 = st.columns([5, 1])
        with s1:
            search_input = st.text_input(
                "area_search", value=st.session_state.search_query,
                placeholder="e.g. Katutura, Klein Windhoek, Pioneerspark...",
                label_visibility="collapsed",
            )
        with s2:
            search_btn = st.button("Go", type="primary", use_container_width=True)

        if search_btn and search_input.strip():
            st.session_state.search_query = search_input.strip()
            st.session_state.search_results = geocode_location(search_input.strip())

        results = st.session_state.get("search_results", [])
        if results:
            st.markdown("**Select a location:**")
            for i, res in enumerate(results[:4]):
                if st.button(f"📍 {res['name']}", key=f"loc_{i}", use_container_width=True):
                    st.session_state.map_center = [res["lat"], res["lon"]]
                    st.session_state.map_zoom   = 15
                    st.session_state.search_results = []
                    st.rerun()
        elif search_btn and search_input.strip():
            st.info("No results found. Try: 'Katutura', 'CBD', 'Khomasdal'.")

        st.markdown('<div class="quick-label">Quick jump</div>', unsafe_allow_html=True)
        areas = {
            "🏙️ CBD":            [-22.5597, 17.0832, 15],
            "🌿 Klein Windhoek":  [-22.5502, 17.0985, 15],
            "🏘️ Katutura":       [-22.5393, 17.0555, 14],
            "🏫 Pioneerspark":    [-22.5812, 17.0741, 15],
            "🏢 Olympia":         [-22.5664, 17.0901, 15],
            "🌳 Eros":            [-22.5526, 17.0843, 15],
        }
        chip_cols = st.columns(3)
        for idx, (label, coords) in enumerate(areas.items()):
            with chip_cols[idx % 3]:
                if st.button(label, key=f"chip_{idx}", use_container_width=True):
                    st.session_state.map_center = [coords[0], coords[1]]
                    st.session_state.map_zoom   = coords[2]
                    st.session_state.search_results = []
                    st.rerun()

        st.markdown("---")

        COLOR_MAP = {
            "Free flow": "#22c55e", "Light": "#facc15",
            "Moderate":  "#f97316", "Heavy": "#ef4444", "Severe": "#a855f7",
        }
        m = folium.Map(location=st.session_state.map_center,
                       zoom_start=st.session_state.map_zoom,
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
                <div style="font-family:'Inter','Segoe UI',Arial;font-size:13px;
                            min-width:200px;background:#1e0e05;color:#f0dfc8;
                            padding:14px 16px;border-radius:10px;
                            border:1px solid #5a3018;border-top:3px solid #c49a6c;">
                  <b style="color:#c49a6c;font-size:14px;">{name}</b><br>
                  <hr style="border-color:rgba(196,154,108,0.25);margin:8px 0;">
                  <table style="width:100%;font-size:12px;border-collapse:collapse;">
                    <tr><td style="color:#7a5535;padding:2px 0;">Type</td>
                        <td style="text-align:right;">{row.get('highway','N/A')}</td></tr>
                    <tr><td style="color:#7a5535;padding:2px 0;">Speed limit</td>
                        <td style="text-align:right;">{row.get('speed_limit','N/A')} km/h</td></tr>
                    <tr><td style="color:#7a5535;padding:2px 0;">AM speed</td>
                        <td style="text-align:right;">{row.get('speed_am',0):.1f} km/h</td></tr>
                    <tr><td style="color:#7a5535;padding:2px 0;">PM speed</td>
                        <td style="text-align:right;">{row.get('speed_pm',0):.1f} km/h</td></tr>
                    <tr><td style="color:#7a5535;padding:2px 0;">Severity</td>
                        <td style="text-align:right;color:{color};font-weight:700;">{severity}</td></tr>
                  </table>
                </div>"""

            folium.PolyLine(
                locations=coords, color=color, weight=4, opacity=0.85,
                tooltip=folium.Tooltip(name, style="font-family:Inter;font-size:13px;"),
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

        if st.session_state.map_zoom >= 15:
            folium.CircleMarker(
                location=st.session_state.map_center, radius=12,
                color="#c49a6c", fill=True,
                fill_color="#8b5e3c", fill_opacity=0.75,
                tooltip="📍 Selected area",
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#1e0e05;padding:16px 20px;border-radius:12px;
                    border:1px solid #5a3018;border-top:3px solid #c49a6c;
                    font-family:'Inter','Segoe UI',Arial;font-size:12px;
                    color:#f0dfc8;box-shadow:0 6px 24px rgba(0,0,0,0.7);">
            <b style="color:#c49a6c;font-size:12px;letter-spacing:1px;
                      text-transform:uppercase;">Congestion Level</b>
            <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
              <div><span style='color:#22c55e;font-size:18px;'>&#9644;</span>&nbsp; Free flow</div>
              <div><span style='color:#facc15;font-size:18px;'>&#9644;</span>&nbsp; Light</div>
              <div><span style='color:#f97316;font-size:18px;'>&#9644;</span>&nbsp; Moderate</div>
              <div><span style='color:#ef4444;font-size:18px;'>&#9644;</span>&nbsp; Heavy</div>
              <div><span style='color:#a855f7;font-size:18px;'>&#9644;</span>&nbsp; Severe</div>
            </div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, height=520, width="100%")

    with chart_col:
        st.subheader("🛣️ Breakdown by Road Type")
        road_congestion = (
            edges.assign(highway=edges["highway"].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
            ))
            .groupby("highway")["congestion_avg"].mean()
            .sort_values(ascending=False).head(10).reset_index()
        )
        road_congestion.columns = ["Road type", "Avg congestion"]
        road_congestion["Avg congestion"] = road_congestion["Avg congestion"].round(3)
        st.bar_chart(road_congestion.set_index("Road type"), height=240, color="#8b5e3c")

        st.subheader("🌅 AM vs PM Peak")
        peak_df = pd.DataFrame({
            "AM Peak": [edges["congestion_am"].mean()],
            "PM Peak": [edges["congestion_pm"].mean()],
        })
        st.bar_chart(peak_df, height=160, color=["#8b5e3c", "#c49a6c"])

        st.subheader("📊 Severity Distribution")
        severity_order = ["Free flow", "Light", "Moderate", "Heavy", "Severe"]
        sev_df = (
            edges["severity"].value_counts()
            .reindex(severity_order).fillna(0).reset_index()
        )
        sev_df.columns = ["Severity", "Count"]
        st.bar_chart(sev_df.set_index("Severity"), height=200, color="#c49a6c")

    st.divider()

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
    st.markdown("""
    <div class="welcome-banner">
        <h2>👋 Welcome to the Windhoek Traffic Dashboard</h2>
        <p>Configure your peak hours in the sidebar, then click
           <strong>▶ Run Analysis</strong> to generate a live congestion map.</p>
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
