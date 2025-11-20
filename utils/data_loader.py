import geopandas as gpd
import streamlit as st

@st.cache_data
def load_geojson(path):
    return gpd.read_file(path)
