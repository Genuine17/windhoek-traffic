"""
Windhoek Traffic Congestion Dashboard
======================================
Streamlit app that wraps the analysis pipeline with an interactive UI.
Theme: Warm Earth & Brown — Natural / Grounded
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

# ── Design tokens ─────────────────────────────────────────────────────────────
# Primary:  #2c1a0e  (deep espresso)
# Accent:   #8b5e3c  (warm brown)
# Accent 2: #c49a6c  (sand/caramel)
# Surface:  #3d2310  (card/panel)
# Text:     #f5e6d3  (warm cream)

st.markdown("""
<style>
  /* ── Viewport meta for mobile ── */
  @viewport { width=device-width; initial-scale=1; }

  /* ── Global ── */
  .stApp {
      background-color: #1a0f07;
      background-image:
          radial-gradient(ellipse at 20% 20%, rgba(196,154,108,0.07) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 80%, rgba(139,94,60,0.06) 0%, transparent 50%);
      color: #f5e6d3;
      font-family: 'Segoe UI', system-ui, sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background-color: #1f0f05;
      border-right: 2px solid #8b5e3c;
  }
  [data-testid="stSidebar"] * { color: #e8d0b0 !important; }
  [data-testid="stSidebar"] .stSlider > div > div > div {
      background: #c49a6c !important;
  }
  [data-testid="stSidebar"] input[type="number"],
  [data-testid="stSidebar"] .stNumberInput input {
      background-color: #2c1a0e !important;
      color: #f5e6d3 !important;
      border: 1px solid #8b5e3c !important;
      border-radius: 6px !important;
  }
  [data-testid="stSidebar"] .stNumberInput button {
      background-color: #2c1a0e !important;
      color: #c49a6c !important;
      border: 1px solid #8b5e3c !important;
  }
  [data-testid="stSidebar"] .stNumberInput button:hover {
      background-color: #8b5e3c !important;
      color: #ffffff !important;
  }

  /* ── Header banner ── */
  .dashboard-header {
      background: linear-gradient(90deg, #2c1a0e 0%, #8b5e3c 50%, #c49a6c 100%);
      padding: 24px 36px;
      border-radius: 14px;
      margin-bottom: 28px;
      box-shadow: 0 4px 28px rgba(196,154,108,0.25);
      border: 1px solid rgba(196,154,108,0.2);
  }
  .dashboard-header h1 {
      color: #ffffff !important;
      font-size: 2rem;
      font-weight: 700;
      margin: 0;
      letter-spacing: 0.4px;
  }
  .dashboard-header p {
      color: #f0d9bc !important;
      font-size: 0.93rem;
      margin: 7px 0 0 0;
  }

  /* ── Subheaders ── */
  h2, h3 {
      color: #c49a6c !important;
      border-bottom: 2px solid rgba(196,154,108,0.25);
      padding-bottom: 5px;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
      background: #f5efe8;
      border: none;
      border-left: 4px solid #8b5e3c;
      border-radius: 10px;
      padding: 16px 18px;
      box-shadow: 0 2px 14px rgba(0,0,0,0.35);
      transition: background 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
      cursor: default;
  }
  [data-testid="stMetric"]:hover {
      background: #8b5e3c;
      box-shadow: 0 6px 24px rgba(139,94,60,0.45);
  }
  [data-testid="stMetric"]:hover [data-testid="stMetricLabel"],
  [data-testid="stMetric"]:hover [data-testid="stMetricValue"],
  [data-testid="stMetric"]:hover [data-testid="stMetricDelta"] {
      color: #ffffff !important;
  }
  [data-testid="stMetricLabel"] {
      color: #6b4226 !important;
      font-weight: 700;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      white-space: normal !important;
      overflow: visible !important;
      text-overflow: unset !important;
      line-height: 1.3;
      transition: color 0.3s ease;
  }
  [data-testid="stMetricValue"] {
      color: #2c1a0e !important;
      font-weight: 800;
      font-size: clamp(1rem, 2vw, 1.6rem);
      white-space: normal !important;
      overflow: visible !important;
      text-overflow: unset !important;
      transition: color 0.3s ease;
  }
  [data-testid="stMetricDelta"] {
      color: #8b5e3c !important;
      font-weight: 600;
      font-size: 0.78rem;
      transition: color 0.3s ease;
  }

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {
      background: linear-gradient(90deg, #2c1a0e, #8b5e3c) !important;
      color: #ffffff !important;
      border: 1px solid #c49a6c !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      letter-spacing: 0.5px;
      box-shadow: 0 3px 14px rgba(196,154,108,0.3);
      transition: all 0.25s ease;
  }
  .stButton > button[kind="primary"]:hover {
      background: linear-gradient(90deg, #8b5e3c, #c49a6c) !important;
      box-shadow: 0 5px 20px rgba(196,154,108,0.5);
      transform: translateY(-1px);
  }

  /* ── Secondary button ── */
  .stButton > button[kind="secondary"] {
      background: transparent !important;
      color: #c49a6c !important;
      border: 1px solid #8b5e3c !important;
      border-radius: 8px !important;
  }
  .stButton > button[kind="secondary"]:hover {
      background: rgba(139,94,60,0.15) !important;
  }

  /* ── Divider ── */
  hr { border-color: rgba(196,154,108,0.2) !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] {
      background: rgba(196,154,108,0.05);
      border: 1px solid rgba(196,154,108,0.2);
      border-radius: 10px;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
      border: 1px solid rgba(196,154,108,0.25) !important;
      border-radius: 8px !important;
  }

  /* ── Alert / info / success ── */
  [data-testid="stAlert"] {
      background: rgba(139,94,60,0.15) !important;
      border-left: 4px solid #c49a6c !important;
      color: #f5e6d3 !important;
      border-radius: 8px;
  }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #c49a6c !important; }

  /* ── Search box styling ── */
  .search-container {
      background: #2c1a0e;
      border: 1px solid #8b5e3c;
      border-radius: 12px;
      padding: 18px 22px;
      margin-bottom: 16px;
  }
  .search-container h4 {
      color: #c49a6c !important;
      border: none !important;
      padding: 0 !important;
      margin: 0 0 10px 0;
      font-size: 1rem;
  }

  /* ── Fact grid ── */
  .fact-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 8px;
  }
  .fact-card {
      flex: 1 1 260px;
      background: #2c1a0e;
      border: 1px solid rgba(196,154,108,0.2);
      border-top: 3px solid #8b5e3c;
      border-radius: 12px;
      padding: 20px 22px;
      box-sizing: border-box;
      transition: background 0.3s ease, border-top-color 0.3s ease, box-shadow 0.3s ease;
  }
  .fact-card:hover {
      background: #8b5e3c;
      border-top-color: #c49a6c;
      box-shadow: 0 8px 28px rgba(196,154,108,0.3);
  }
  .fact-card:hover .fact-label { color: #f5e6d3 !important; }
  .fact-card:hover .fact-value { color: #ffffff !important; }
  .fact-card:hover .fact-desc  { color: #fde8c8 !important; opacity: 1; }
  .fact-icon  { font-size: 2rem; margin-bottom: 10px; }
  .fact-label { color: #c49a6c; font-weight: 700; font-size: 0.88rem;
                text-transform: uppercase; letter-spacing: 0.6px;
                margin-bottom: 4px; transition: color 0.3s ease; }
  .fact-value { color: #ffffff; font-size: 1.65rem; font-weight: 800;
                margin-bottom: 6px; transition: color 0.3s ease; }
  .fact-desc  { color: #b89070; font-size: 0.84rem; line-height: 1.5;
                opacity: 0.85; transition: color 0.3s ease, opacity 0.3s ease; }

  /* ── Welcome banner ── */
  .welcome-banner {
      background: linear-gradient(135deg, rgba(196,154,108,0.1), rgba(139,94,60,0.1));
      border: 1px solid rgba(196,154,108,0.25);
      border-radius: 14px;
      padding: 28px 36px;
      margin-bottom: 28px;
      text-align: center;
  }
  .welcome-banner h2 { color: #c49a6c !important; border: none !important; }
  .welcome-banner p  { color: #d4b896; font-size: 1.05rem; margin-top: 8px; }

  /* ── Search result chip ── */
  .search-result-chip {
      display: inline-block;
      background: rgba(139,94,60,0.2);
      border: 1px solid #8b5e3c;
      border-radius: 20px;
      padding: 4px 14px;
      color: #c49a6c;
      font-size: 0.82rem;
      margin: 4px 4px 4px 0;
  }

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


def _hash_geodataframe(gdf):
    """Deterministic hash for a GeoDataFrame so st.cache_data can key on it."""
    h = hashlib.md5()
    h.update(str(gdf.shape).encode())
    try:
        h.update(gdf.geometry.to_wkt().str.cat().encode())
    except Exception:
        h.update(str(gdf.index.tolist()).encode())
    return h.hexdigest()


# ── Cache fix: wrap simulate_gps_observations with a hashable signature ───────
@st.cache_data(
    hash_funcs={"geopandas.geodataframe.GeoDataFrame": _hash_geodataframe}
)
def cached_simulate(edges, seed: int):
    return simulate_gps_observations(edges, seed=seed)


# ── Nominatim geocoding helper ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def geocode_location(query: str):
    """
    Search Windhoek for a suburb/area using Nominatim.
    Returns list of dicts with name, lat, lon.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{query}, Windhoek, Namibia",
            "format": "json",
            "limit": 5,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "WindhoekTrafficDashboard/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=6)
        results = r.json()
        return [
            {
                "name": res.get("display_name", "").split(",")[0],
                "full": res.get("display_name", ""),
                "lat": float(res["lat"]),
                "lon": float(res["lon"]),
            }
            for res in results
        ]
    except Exception:
        return []


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
if "map_center" not in st.session_state:
    st.session_state.map_center = [-22.5597, 17.0832]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 13
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching road network from OpenStreetMap..."):
        edges = fetch_road_network()
    with st.spinner("Simulating peak-hour speeds..."):
        edges = cached_simulate(edges, seed=int(seed))          # ← cache fix
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
    c1.metric("Segments", f"{total:,}")
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

        # ── Area Search ───────────────────────────────────────────────────────
        st.markdown('<div class="search-container"><h4>🔍 Search Area</h4></div>',
                    unsafe_allow_html=True)

        search_cols = st.columns([5, 1])
        with search_cols[0]:
            search_input = st.text_input(
                "search_input",
                value=st.session_state.search_query,
                placeholder="e.g. Katutura, Klein Windhoek, Pioneerspark...",
                label_visibility="collapsed",
            )
        with search_cols[1]:
            search_btn = st.button("Search", type="primary", use_container_width=True)

        # Trigger search
        if search_btn and search_input.strip():
            st.session_state.search_query = search_input.strip()
            results = geocode_location(search_input.strip())
            st.session_state.search_results = results
        elif "search_results" not in st.session_state:
            st.session_state.search_results = []

        # Show results and let user pick
        results = st.session_state.get("search_results", [])
        if results:
            st.markdown("**Select a location to fly to:**")
            for i, res in enumerate(results[:4]):
                btn_label = f"📍 {res['name']}"
                if st.button(btn_label, key=f"loc_{i}", use_container_width=True):
                    st.session_state.map_center = [res["lat"], res["lon"]]
                    st.session_state.map_zoom = 15
                    st.session_state.search_results = []
                    st.rerun()
        elif search_btn and search_input.strip():
            st.info("No locations found. Try a different name (e.g. 'Katutura', 'CBD').")

        # Quick-jump chips for popular areas
        st.markdown("**Quick jump:**")
        areas = {
            "🏙️ CBD":           [-22.5597, 17.0832, 15],
            "🌿 Klein Windhoek": [-22.5502, 17.0985, 15],
            "🏘️ Katutura":      [-22.5393, 17.0555, 14],
            "🏫 Pioneerspark":   [-22.5812, 17.0741, 15],
            "🏢 Olympia":        [-22.5664, 17.0901, 15],
            "🌳 Eros":           [-22.5526, 17.0843, 15],
        }
        chip_cols = st.columns(3)
        for idx, (label, coords) in enumerate(areas.items()):
            with chip_cols[idx % 3]:
                if st.button(label, key=f"chip_{idx}", use_container_width=True):
                    st.session_state.map_center = [coords[0], coords[1]]
                    st.session_state.map_zoom = coords[2]
                    st.session_state.search_results = []
                    st.rerun()

        st.markdown("---")

        # ── Build & render map ────────────────────────────────────────────────
        COLOR_MAP = {
            "Free flow": "#22c55e",
            "Light":     "#facc15",
            "Moderate":  "#f97316",
            "Heavy":     "#ef4444",
            "Severe":    "#a855f7",
        }

        m = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
            tiles="CartoDB dark_matter",
        )

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
                            background:#2c1a0e;color:#f5e6d3;padding:12px 14px;border-radius:8px;
                            border:1px solid #8b5e3c;">
                <b style="color:#c49a6c;font-size:14px;">{name}</b><br>
                <hr style="border-color:rgba(196,154,108,0.3);margin:6px 0;">
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

        # Add a marker if user jumped to a searched location
        if st.session_state.map_zoom >= 15:
            folium.CircleMarker(
                location=st.session_state.map_center,
                radius=10,
                color="#c49a6c",
                fill=True,
                fill_color="#8b5e3c",
                fill_opacity=0.7,
                tooltip="📍 Selected area",
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#2c1a0e;padding:14px 18px;border-radius:10px;
                    border:1px solid #8b5e3c;font-family:'Segoe UI',Arial;font-size:13px;
                    color:#f5e6d3;box-shadow:0 4px 18px rgba(0,0,0,0.6);">
            <b style="color:#c49a6c;font-size:14px;">Congestion Level</b><br><br>
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
        st.bar_chart(road_congestion.set_index("Road type"), height=240, color="#8b5e3c")

        st.subheader("🌊 AM vs PM Peak")
        peak_df = pd.DataFrame({
            "AM Peak": [edges["congestion_am"].mean()],
            "PM Peak": [edges["congestion_pm"].mean()],
        })
        st.bar_chart(peak_df, height=160, color=["#8b5e3c", "#c49a6c"])

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
        st.bar_chart(sev_df.set_index("Severity"), height=200, color="#c49a6c")

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
           <b style="color:#c49a6c;">▶ Run Analysis</b>
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
