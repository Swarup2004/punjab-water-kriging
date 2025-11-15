# -------------------------------------------------------------
# Punjab River Pollutant Visualizer — Stable + Upstream Tracing
# PCA-based flow direction, synthetic river polyline, NO glitches
# -------------------------------------------------------------

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
DATA_PATH = "./data/river_data_cleaned.csv"
GEOJSON_PATH = "assets/punjab_districts.geojson"

MAP_CENTER = [30.9, 75.8]
MAP_ZOOM = 7
PUNJAB_BBOX = [28.5, 73.5, 32.5, 77.5]

st.set_page_config(layout="wide", page_title="Punjab River Visualizer — Stable + Flow")

# -------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize lat/lon names
    latc = [c for c in df.columns if c.lower() in ("lat", "latitude")]
    lonc = [c for c in df.columns if c.lower() in ("lon", "longitude", "long")]

    if latc: df = df.rename(columns={latc[0]: "lat"})
    if lonc: df = df.rename(columns={lonc[0]: "lon"})

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Assign site_id if missing
    if "site_id" not in df.columns:
        df["site_id"] = df.index.astype(str)

    return df


@st.cache_data
def load_geojson(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return gpd.read_file(path)
    except:
        return None


# -------------------------------------------------------------
# POLLUTANT DETECTORS
# -------------------------------------------------------------
def detect_pollutants(df):
    bases = set()
    for col in df.columns:
        for m in ("Nov", "Dec", "Jan"):
            if col.endswith(f"_{m}"):
                bases.add(col.replace(f"_{m}", ""))
    return sorted(list(bases))


def find_month_cols(df, base):
    result = []
    for m in ("Nov", "Dec", "Jan"):
        col = f"{base}_{m}"
        if col in df.columns:
            result.append((col, m))
    return result


def compute_aggregate(df, base, months):
    cols = [f"{base}_{m}" for m in months if f"{base}_{m}" in df.columns]
    if not cols:
        return pd.Series([0]*len(df), index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.max(axis=1).fillna(0)


# -------------------------------------------------------------
# PCA FLOW + SYNTHETIC RIVER
# -------------------------------------------------------------
def compute_pca_order(df):
    """Return site indices sorted along principal river axis."""
    coords = df[["lat", "lon"]].values
    pca = PCA(n_components=1)
    proj = pca.fit_transform(coords)
    return np.argsort(proj[:, 0])


def generate_smooth_river(df, order):
    """Create smooth polyline from sorted site coordinates."""
    pts = df.loc[order, ["lat", "lon"]].values

    # Simple smoothing: moving average (can be replaced by spline)
    smoothed = []
    for i in range(len(pts)):
        w_start = max(0, i-2)
        w_end = min(len(pts), i+3)
        smoothed.append(pts[w_start:w_end].mean(axis=0))

    return np.array(smoothed)


def add_flow_arrows(map_obj, coords, color="#0077ff"):
    """Add arrow markers along synthetic river."""
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color=color, weight=3, opacity=0.9
        ).add_to(map_obj)

        # Add small directional arrow
        folium.RegularPolygonMarker(
            location=[lat2, lon2],
            number_of_sides=3,
            radius=6,
            rotation=45,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(map_obj)


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
df = load_data(DATA_PATH)
gdf = load_geojson(GEOJSON_PATH)
pollutant_bases = detect_pollutants(df)
months_all = ["Nov", "Dec", "Jan"]

if not pollutant_bases:
    st.error("No pollutant columns detected. Ensure format: Pollutant_Nov, Pollutant_Dec, Pollutant_Jan")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.title("Controls")

pollutant = st.sidebar.selectbox("Pollutant", pollutant_bases)
months_selected = st.sidebar.multiselect("Months", months_all, default=months_all)

show_districts = st.sidebar.checkbox("Show districts", True)
cluster_markers = st.sidebar.checkbox("Cluster markers", True)
use_diff = st.sidebar.checkbox("Downstream differential", False)
diff_mode = st.sidebar.selectbox("Mode", ["signed", "positive-only"])

cmap_choice = st.sidebar.selectbox("Colormap", ["RdBu", "coolwarm", "PRGn", "PiYG", "seismic"])


# -------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------
map_col, right_col = st.columns([2.5, 1])

with map_col:
    st.header("Map")

    # ✔ SAFE — no width/height here
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, control_scale=True)

    # Fit to bounding box
    m.fit_bounds([[PUNJAB_BBOX[0], PUNJAB_BBOX[1]], [PUNJAB_BBOX[2], PUNJAB_BBOX[3]]])

    # District overlay
    if gdf is not None and show_districts:
        folium.GeoJson(
            gdf,
            name="districts",
            style_function=lambda feat: {"fillColor":"#00000000", "color":"#555", "weight":1}
        ).add_to(m)

    # PCA ordering + synthetic river
    order = compute_pca_order(df)
    river_line = generate_smooth_river(df, order)
    add_flow_arrows(m, river_line)

    # Compute pollutant values
    agg = compute_aggregate(df, pollutant, months_selected)

    # Differential if enabled
    if use_diff:
        diffs = pd.Series(0, index=df.index)
        for i in range(1, len(order)):
            cur, prev = order[i], order[i-1]
            diffs[cur] = agg[cur] - agg[prev]
        if diff_mode == "positive-only":
            diffs = diffs.clip(lower=0)
        values = diffs
    else:
        values = agg

    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax: vmax = vmin + 1

    cmap = mpl.cm.get_cmap(cmap_choice)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if use_diff else mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    def colorize(v):
        return mcolors.to_hex(sm.to_rgba(float(v)))

    # Marker cluster
    cluster = MarkerCluster() if cluster_markers else m
    if cluster_markers: m.add_child(cluster)

    # Add markers
    for idx, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        val = values[idx]
        color = colorize(val)

        folium.CircleMarker(
            location=(lat, lon),
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"{row['site_id']}<br>{pollutant}: {val:.2f}"
        ).add_to(cluster)

    map_data = st_folium(m, height=600, use_container_width=True)


# -------------------------------------------------------------
# RIGHT PANEL — site details + upstream tracing
# -------------------------------------------------------------
with right_col:
    st.header("Selected Site")

    last_click = map_data.get("last_clicked")
    selected_idx = None

    if last_click:
        latc, lonc = last_click["lat"], last_click["lng"]
        dists = (df["lat"] - latc)**2 + (df["lon"] - lonc)**2
        if dists.min() < 0.0004:
            selected_idx = int(dists.idxmin())

    if selected_idx is None:
        st.info("Click any marker on the map.")
        st.stop()

    row = df.loc[selected_idx]
    st.subheader(row["site_id"])
    st.write(f"Lat: {row['lat']}, Lon: {row['lon']}")

    # Upstream tracing
    st.markdown("### Upstream Contributors")

    pos = list(order).index(selected_idx)
    upstream = order[:pos]
    downstream = order[pos+1:]

    st.write(f"Upstream sites: {len(upstream)}")
    st.write(f"Downstream sites: {len(downstream)}")

    # Pollutant trends
    pairs = find_month_cols(df, pollutant)
    if pairs:
        months = [m for _, m in pairs]
        vals = [row[c] for c, _ in pairs]

        fig = go.Figure()
        fig.add_bar(x=months, y=vals)
        fig.add_scatter(x=months, y=vals, mode="lines+markers")
        fig.update_layout(title=f"{pollutant} over months", height=300)
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------
# END
# -------------------------------------------------------------
st.markdown("---")
st.markdown("Built with automatic PCA-based flow, upstream tracing, synthetic river curves, and stable non-glitching Folium rendering.")
