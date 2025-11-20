import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium

# local imports
from utils.map_layers import add_river_polyline
from utils.flow_arrows import add_flow_arrows
from utils.heatmap import add_heatmap
from utils.trace import show_trace
from utils.industry_hotspots import add_hotspot_layer
from utils.data_loader import load_geojson
from utils.colors import pollutant_color


# -----------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------
st.set_page_config(page_title="HydroKriging Pollution Viewer", layout="wide")
st.title("🌊 HydroKriging River Pollution Visualizer")


# -----------------------------------------------------
# Load data
# -----------------------------------------------------
gdf = load_geojson("data/hydrokriging_predictions.geojson")

pollutant_cols = [
    c for c in gdf.columns
    if c not in ["lat","lon","geometry","elevation","slope_to_next",
                 "flow_direction_deg","distance_m","site_id","river_name"]
]

try:
    river_line = gpd.read_file("data/river_polyline.geojson")
except:
    river_line = None


# -----------------------------------------------------
# Sidebar UI
# -----------------------------------------------------
st.sidebar.header("⚙ Controls")

selected_pollutant = st.sidebar.selectbox("Select pollutant", pollutant_cols)

SHOW_POINTS = st.sidebar.checkbox("Show kriged points", True)
SHOW_HEATMAP = st.sidebar.checkbox("Show heatmap layer", True)
SHOW_FLOW = st.sidebar.checkbox("Show flow arrows", True)
SHOW_RIVER = st.sidebar.checkbox("Show river polyline", True)
SHOW_TRACE = st.sidebar.checkbox("Trace upstream/downstream", False)
SHOW_HOTSPOTS = st.sidebar.checkbox("Predict industry hotspots", False)

point_size = st.sidebar.slider("Point size", 3, 15, 7)
opacity = st.sidebar.slider("Heatmap opacity", 0.1, 1.0, 0.7)


# -----------------------------------------------------
# Create Map
# -----------------------------------------------------
center = [gdf.lat.mean(), gdf.lon.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

if SHOW_RIVER and river_line is not None:
    add_river_polyline(m, river_line)

if SHOW_POINTS:
    for _, row in gdf.iterrows():
        val = row[selected_pollutant]
        col = pollutant_color(val, gdf[selected_pollutant])
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=point_size,
            color=col,
            fill=True, fill_color=col, fill_opacity=0.9,
            tooltip=f"{selected_pollutant}: {val:.2f}"
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
# Render map
# -----------------------------------------------------
st.subheader(f"Pollutant: **{selected_pollutant}**")
st_folium(m, height=680, width=1450)


# -----------------------------------------------------
# Table + downloads
# -----------------------------------------------------
st.subheader("📄 Dataset")
st.dataframe(gdf.head(500))

st.download_button(
    "Download as CSV",
    gdf.to_csv(index=False).encode("utf-8"),
    "hydrokriging_predictions.csv",
    "text/csv"
)
