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
# STREAMLIT PAGE SETUP
# -----------------------------------------------------
st.set_page_config(
    page_title="HydroKriging Pollution Visualizer",
    layout="wide"
)
st.title("🌊 HydroKriging River Pollution Visualizer")


# -----------------------------------------------------
# CACHED DATA LOADERS
# -----------------------------------------------------
@st.cache_resource
def load_predictions():
    return load_geojson("data/hydrokriging_predictions.geojson")

@st.cache_resource
def load_river():
    try:
        return gpd.read_file("data/river_polyline.geojson")
    except:
        return None


gdf = load_predictions()
river_line = load_river()


# -----------------------------------------------------
# POLLUTANT COLUMNS
# -----------------------------------------------------
EXCLUDE_COLS = [
    "lat", "lon", "geometry", "elevation", "slope_to_next",
    "flow_direction_deg", "distance_m", "site_id", "river_name"
]

pollutant_cols = [c for c in gdf.columns if c not in EXCLUDE_COLS]


# -----------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------
st.sidebar.header("⚙ Controls")

selected_pollutant = st.sidebar.selectbox("Select pollutant", pollutant_cols)

SHOW_POINTS = st.sidebar.checkbox("Show kriged points", True)
SHOW_HEATMAP = st.sidebar.checkbox("Show heatmap layer", True)
SHOW_FLOW = st.sidebar.checkbox("Show flow arrows", False)
SHOW_RIVER = st.sidebar.checkbox("Show river polyline", True)
SHOW_TRACE = st.sidebar.checkbox("Trace upstream/downstream", False)
SHOW_HOTSPOTS = st.sidebar.checkbox("Predict industry hotspots", False)

point_size = st.sidebar.slider("Point size", 3, 15, 7)
opacity = st.sidebar.slider("Heatmap opacity", 0.1, 1.0, 0.6)


# -----------------------------------------------------
# PRECOMPUTE COLORS (performance boost)
# -----------------------------------------------------
vals = gdf[selected_pollutant]
colors = [pollutant_color(v, vals) for v in vals]


# -----------------------------------------------------
# BASE MAP (canvas accelerated)
# -----------------------------------------------------
center = [gdf.lat.mean(), gdf.lon.mean()]

m = folium.Map(
    location=center,
    zoom_start=13,
    tiles="CartoDB Positron",
    prefer_canvas=True
)


# -----------------------------------------------------
# MAP LAYERS
# -----------------------------------------------------

if SHOW_RIVER and river_line is not None:
    add_river_polyline(m, river_line)

if SHOW_POINTS:
    lat_arr = gdf.lat.values
    lon_arr = gdf.lon.values

    for i in range(len(gdf)):
        folium.CircleMarker(
            location=[lat_arr[i], lon_arr[i]],
            radius=point_size,
            color=colors[i],
            fill=True,
            fill_color=colors[i],
            fill_opacity=0.9,
            tooltip=f"{selected_pollutant}: {vals.iloc[i]:.2f}"
        ).add_to(m)

if SHOW_HEATMAP:
    add_heatmap(m, gdf, selected_pollutant, opacity)

if SHOW_FLOW:
    add_flow_arrows(m, gdf)

if SHOW_HOTSPOTS:
    add_hotspot_layer(m, gdf, selected_pollutant)

if SHOW_TRACE:
    show_trace(m, gdf)


# -----------------------------------------------------
# RENDER MAP (NO unsupported args)
# -----------------------------------------------------
st.subheader(f"Pollutant: **{selected_pollutant}**")

st_folium(
    m,
    height=700,
    width=1500,
    key="map"     # ✔ this prevents redraw flicker
)


# -----------------------------------------------------
# DOWNLOAD BUTTON ONLY (NO TABLE)
# -----------------------------------------------------
st.download_button(
    "Download dataset (CSV)",
    gdf.to_csv(index=False).encode("utf-8"),
    "hydrokriging_predictions.csv",
    "text/csv"
)
