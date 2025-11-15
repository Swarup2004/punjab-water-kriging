# -------------------------------------------------------------
# Punjab River Pollutant Visualizer — Stable + PCA Flow + Upstream
# Fully fixed: pollutant detection restored, map glitch removed
# -------------------------------------------------------------

import os
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Punjab River Visualizer — Stable + Flow")

DATA_PATH = "./data/river_data_cleaned.csv"
GEOJSON_PATH = "assets/punjab_districts.geojson"

MAP_CENTER = [30.9, 75.8]
MAP_ZOOM = 7
PUNJAB_BBOX = [28.5, 73.5, 32.5, 77.5]

# -------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Month-name normalization
    replace_map = {
        "Nov-24_": "Nov_", "Dec-24_": "Dec_", "Jan-25_": "Jan_",
        "Nov-24": "Nov", "Dec-24": "Dec", "Jan-25": "Jan"
    }
    for old, new in replace_map.items():
        df.columns = [c.replace(old, new) for c in df.columns]

    # lat/lon
    latc = [c for c in df.columns if c.lower() in ("lat", "latitude", "latitude (deg)")]
    lonc = [c for c in df.columns if c.lower() in ("lon", "longitude", "long", "longitude (deg)")]

    if latc: df = df.rename(columns={latc[0]: "lat"})
    if lonc: df = df.rename(columns={lonc[0]: "lon"})

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # site ID
    site_candidates = [c for c in df.columns if "site" in c.lower() or "location" in c.lower()]
    if site_candidates:
        df = df.rename(columns={site_candidates[0]: "site_id"})
    else:
        df["site_id"] = df.index.astype(str)

    return df


@st.cache_data
def load_geojson(path):
    if not os.path.exists(path):
        return None
    try:
        return gpd.read_file(path)
    except:
        return None


# -------------------------------------------------------------
# POLLUTANT DETECTION (RESTORED FROM YOUR WORKING VERSION)
# -------------------------------------------------------------

def detect_pollutants(df: pd.DataFrame) -> List[str]:
    """
    Detect pollutant bases supporting BOTH:
      - Nov_pH
      - pH_Nov
    """

    months = ["Nov", "Dec", "Jan"]
    pollutant_bases = set()

    for col in df.columns:
        for m in months:
            # prefix
            if col.startswith(f"{m}_"):
                pollutant_bases.add(col.split(f"{m}_", 1)[1])

            # suffix
            if col.endswith(f"_{m}"):
                pollutant_bases.add(col.rsplit(f"_{m}", 1)[0])

    # fallback
    if not pollutant_bases:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pollutant_bases = set([c for c in numeric_cols if c not in ("lat", "lon")][:12])

    return sorted(pollutant_bases)


def find_month_cols(df: pd.DataFrame, base: str):
    pairs = []
    for m in ["Nov", "Dec", "Jan"]:
        pref = f"{m}_{base}"      # Nov_pH
        suff = f"{base}_{m}"      # pH_Nov

        if pref in df.columns:
            pairs.append((pref, m))

        elif suff in df.columns:
            pairs.append((suff, m))

        else:
            # case-insensitive rescue
            for c in df.columns:
                if c.lower() == pref.lower() or c.lower() == suff.lower():
                    pairs.append((c, m))
                    break
    return pairs


def compute_aggregate(df, base, months):
    pairs = find_month_cols(df, base)
    cols = [c for c, m in pairs if m in months]
    if not cols:
        return pd.Series([0]*len(df), index=df.index)

    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.max(axis=1).fillna(0)


# -------------------------------------------------------------
# PCA ORDER + SYNTHETIC RIVER LINE
# -------------------------------------------------------------

def compute_pca_order(df):
    coords = df[["lat", "lon"]].values
    pca = PCA(n_components=1)
    proj = pca.fit_transform(coords)
    return np.argsort(proj[:, 0])


def smooth_line(points: np.ndarray):
    smoothed = []
    n = len(points)
    for i in range(n):
        w_start = max(0, i-2)
        w_end = min(n, i+3)
        smoothed.append(points[w_start:w_end].mean(axis=0))
    return np.array(smoothed)


def draw_flow_line(fmap, coords, color="#1f3cff"):
    for i in range(1, len(coords)):
        p1 = coords[i-1]
        p2 = coords[i]

        folium.PolyLine([tuple(p1), tuple(p2)], color=color, weight=4, opacity=0.9).add_to(fmap)

        # directional arrow
        folium.RegularPolygonMarker(
            location=tuple(p2),
            number_of_sides=3,
            radius=6,
            rotation=45,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(fmap)


# -------------------------------------------------------------
# LOAD
# -------------------------------------------------------------

if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)
gdf = load_geojson(GEOJSON_PATH)
pollutant_bases = detect_pollutants(df)

if not pollutant_bases:
    st.error("No pollutant columns detected. Ensure format: Pollutant_Nov / Nov_Pollutant")
    st.stop()

months_all = ["Nov", "Dec", "Jan"]

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------

st.sidebar.header("Controls")

pollutant = st.sidebar.selectbox("Pollutant", pollutant_bases)
months_selected = st.sidebar.multiselect("Months", months_all, default=months_all)

differential = st.sidebar.checkbox("Downstream differential", False)
diff_mode = st.sidebar.selectbox("Mode", ["signed", "positive-only"])

cmap_choice = st.sidebar.selectbox("Colormap", ["RdBu", "coolwarm", "seismic", "PiYG", "PRGn"])

cluster_markers = st.sidebar.checkbox("Cluster markers", True)
show_districts = st.sidebar.checkbox("Show districts", True)
search_text = st.sidebar.text_input("Search site name…")

# -------------------------------------------------------------
# LAYOUT
# -------------------------------------------------------------

map_col, right_col = st.columns([2.5, 1])

with map_col:
    st.header("Map")

    # ✔ DO NOT SET HEIGHT HERE
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, control_scale=True)

    # bounding box
    m.fit_bounds([[PUNJAB_BBOX[0], PUNJAB_BBOX[1]], [PUNJAB_BBOX[2], PUNJAB_BBOX[3]]])

    # district layer
    if gdf is not None and show_districts:
        folium.GeoJson(
            gdf,
            style_function=lambda f: {"color": "#555", "weight": 1, "fillColor": "#00000000"}
        ).add_to(m)

    # PCA ordering + river
    order = compute_pca_order(df)
    raw_points = df.loc[order, ["lat", "lon"]].values
    smoothed = smooth_line(raw_points)
    draw_flow_line(m, smoothed)

    # Values
    agg = compute_aggregate(df, pollutant, months_selected)

    if differential:
        diffs = pd.Series(0.0, index=df.index)
        for i in range(1, len(order)):
            cur = order[i]
            prev = order[i-1]
            diffs[cur] = agg[cur] - agg[prev]
        if diff_mode == "positive-only":
            diffs = diffs.clip(lower=0)
        values = diffs
    else:
        values = agg

    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax: vmax = vmin + 1

    cmap = mpl.cm.get_cmap(cmap_choice)
    norm = (
        mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        if differential
        else mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    def colorize(v):
        return mcolors.to_hex(sm.to_rgba(float(v)))

    # clustering
    cluster_layer = MarkerCluster() if cluster_markers else m
    if cluster_markers: m.add_child(cluster_layer)

    # search filtering
    df_filtered = df.copy()
    if search_text.strip():
        df_filtered = df[df["site_id"].astype(str).str.contains(search_text, case=False)]

    # markers
    for idx, row in df_filtered.iterrows():
        val = values[idx]
        color = colorize(val)

        folium.CircleMarker(
            (row["lat"], row["lon"]),
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"<b>{row['site_id']}</b><br>{pollutant}: {val:.2f}"
        ).add_to(cluster_layer)

    map_data = st_folium(m, height=600, use_container_width=True, returned_objects=["last_clicked"])

# -------------------------------------------------------------
# RIGHT SIDEBAR (DETAILS)
# -------------------------------------------------------------

with right_col:
    st.header("Selected Site")

    last_click = map_data.get("last_clicked")
    selected_idx = None

    if last_click:
        lat_click, lon_click = last_click["lat"], last_click["lng"]
        dists = (df["lat"] - lat_click)**2 + (df["lon"] - lon_click)**2
        if dists.min() < 0.0005:
            selected_idx = int(dists.idxmin())

    if selected_idx is None:
        st.info("Click a marker to view details.")
        st.stop()

    row = df.loc[selected_idx]
    st.subheader(row["site_id"])
    st.write(f"Lat: {row['lat']}, Lon: {row['lon']}")

    # upstream sites
    idx_pos = list(order).index(selected_idx)
    upstream = order[:idx_pos]
    downstream = order[idx_pos+1:]

    st.markdown("### Flow Position")
    st.write(f"Upstream sites: {len(upstream)}")
    st.write(f"Downstream sites: {len(downstream)}")

    # pollutant chart
    cols = find_month_cols(df, pollutant)
    if cols:
        months = [m for _, m in cols]
        vals = [row[c] for c, _ in cols]

        fig = go.Figure()
        fig.add_bar(x=months, y=vals)
        fig.add_scatter(x=months, y=vals, mode="lines+markers")
        fig.update_layout(title=f"{pollutant} across months", height=300)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("✔ PCA flow • ✔ Upstream tracing • ✔ Stable map • ✔ Prefix + suffix pollutant support")
