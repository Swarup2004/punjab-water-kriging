import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium

# Local imports
from utils.map_layers import add_river_polyline
from utils.flow_arrows import add_flow_arrows
from utils.heatmap import add_heatmap
from utils.trace import show_trace
from utils.industry_hotspots import add_hotspot_layer
from utils.data_loader import load_geojson
from utils.colors import pollutant_color


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="HydroKriging Pollution Visualizer", layout="wide")
st.title("🌊 HydroKriging River Pollution Visualizer")


# -----------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------
if "gdf" not in st.session_state:
    st.session_state["gdf"] = load_geojson("data/hydrokriging_predictions.geojson")

if "river_line" not in st.session_state:
    try:
        st.session_state["river_line"] = gpd.read_file("data/river_polyline.geojson")
    except:
        st.session_state["river_line"] = None

if "color_cache" not in st.session_state:
    st.session_state["color_cache"] = {}

if "heatmap_cache" not in st.session_state:
    st.session_state["heatmap_cache"] = {}

if "arrow_cache" not in st.session_state:
    st.session_state["arrow_cache"] = None

if "trace_cache" not in st.session_state:
    st.session_state["trace_cache"] = {}

if "hotspot_cache" not in st.session_state:
    st.session_state["hotspot_cache"] = {}

gdf = st.session_state["gdf"]
river_line = st.session_state["river_line"]


# -----------------------------------------------------
# IDENTIFY POLLUTANTS
# -----------------------------------------------------
EXCLUDE_COLS = [
    "lat", "lon", "geometry", "elevation", "slope_to_next",
    "flow_direction_deg", "distance_m", "site_id", "river_name"
]

pollutant_cols = [c for c in gdf.columns if c not in EXCLUDE_COLS]


# -----------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------
st.sidebar.header("⚙ Controls")

selected_pollutant = st.sidebar.selectbox(
    "Select pollutant",
    pollutant_cols
)

SHOW_POINTS = st.sidebar.checkbox("Show kriged points", True)
SHOW_HEATMAP = st.sidebar.checkbox("Show heatmap layer", True)
SHOW_FLOW = st.sidebar.checkbox("Show flow arrows", False)
SHOW_RIVER = st.sidebar.checkbox("Show river polyline", True)
SHOW_TRACE = st.sidebar.checkbox("Trace upstream/downstream", False)
SHOW_HOTSPOTS = st.sidebar.checkbox("Predict industry hotspots", False)

point_size = st.sidebar.slider("Point size", 3, 15, 7)
opacity = st.sidebar.slider("Heatmap opacity", 0.1, 1.0, 0.6)


# -----------------------------------------------------
# PRECOMPUTE COLORS (ONLY ONCE PER POLLUTANT)
# -----------------------------------------------------
if selected_pollutant not in st.session_state["color_cache"]:
    vals = gdf[selected_pollutant]
    st.session_state["color_cache"][selected_pollutant] = [
        pollutant_color(v, vals) for v in vals
    ]

colors = st.session_state["color_cache"][selected_pollutant]


# -----------------------------------------------------
# BASE MAP (NO LAG)
# -----------------------------------------------------
center = [gdf.lat.mean(), gdf.lon.mean()]
m = folium.Map(
    location=center,
    zoom_start=13,
    tiles="CartoDB Positron",
    prefer_canvas=True  # drastically improves rendering speed
)


# -----------------------------------------------------
# MAP LAYERS (ALL CACHED)
# -----------------------------------------------------

# 1. River
if SHOW_RIVER and river_line is not None:
    add_river_polyline(m, river_line)


# 2. Points
if SHOW_POINTS:
    lat = gdf.lat.values
    lon = gdf.lon.values
    vals = gdf[selected_pollutant].values

    for i in range(len(gdf)):
        folium.CircleMarker(
            location=[lat[i], lon[i]],
            radius=point_size,
            color=colors[i],
            fill=True,
            fill_color=colors[i],
            fill_opacity=0.9,
            tooltip=f"{selected_pollutant}: {vals[i]:.2f}"
        ).add_to(m)


# 3. Heatmap (cached)
if SHOW_HEATMAP:
    if selected_pollutant not in st.session_state["heatmap_cache"]:
        st.session_state["heatmap_cache"][selected_pollutant] = True
    add_heatmap(m, gdf, selected_pollutant, opacity)


# 4. Flow arrows (cached)
if SHOW_FLOW:
    if st.session_state["arrow_cache"] is None:
        st.session_state["arrow_cache"] = True
    add_flow_arrows(m, gdf)


# 5. Hotspots (cached)
if SHOW_HOTSPOTS:
    if selected_pollutant not in st.session_state["hotspot_cache"]:
        st.session_state["hotspot_cache"][selected_pollutant] = True
    add_hotspot_layer(m, gdf, selected_pollutant)


# 6. Trace (cached)
if SHOW_TRACE:
    if selected_pollutant not in st.session_state["trace_cache"]:
        st.session_state["trace_cache"][selected_pollutant] = True
    show_trace(m, gdf)


# -----------------------------------------------------
# RENDER MAP — NO STATE RETURN (NO RERUN ON ZOOM)
# -----------------------------------------------------
st.subheader(f"Pollutant: **{selected_pollutant}**")

st_folium(
    m,
    height=700,
    width=1500,
    key="map",
    
)


# -----------------------------------------------------
# DOWNLOAD BUTTON
# -----------------------------------------------------
st.download_button(
    "Download dataset (CSV)",
    gdf.to_csv(index=False).encode("utf-8"),
    "hydrokriging_predictions.csv",
    "text/csv"
)
