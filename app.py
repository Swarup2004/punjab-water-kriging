# -------------------------------------------------------------
# Punjab N-Choe Pollutant Visualizer — Ultra Edition (Clean Build)
# -------------------------------------------------------------
# Features:
#   ✓ Kriging interpolation along synthetic river
#   ✓ Animated flow arrows (PCA-based)
#   ✓ Pollution source detection
#   ✓ Multi-river clustering (DBSCAN)
#   ✓ Month-name normalization
#   ✓ Streamlit Cloud compatible
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import matplotlib as mpl
import matplotlib.colors as mcolors

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Punjab N-Choe Pollutant Visualizer — Ultra")

DATA_PATH = "./data/river_data_cleaned.csv"
GEOJSON_PATH = "assets/punjab_districts.geojson"

MAP_CENTER = [30.70, 76.74]
MAP_ZOOM = 13

# -------------------------------------------------------------
# LOAD CSV + NORMALIZE
# -------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize month labels
    replacements = {
        "Nov-24_": "Nov_", "Dec-24_": "Dec_", "Jan-25_": "Jan_",
        "Nov-24": "Nov", "Dec-24": "Dec", "Jan-25": "Jan"
    }
    for old, new in replacements.items():
        df.columns = [c.replace(old, new) for c in df.columns]

    # Coordinates
    lat_col = next((c for c in df.columns if c.lower().startswith("lat")), None)
    lon_col = next((c for c in df.columns if c.lower().startswith("lon")), None)

    if lat_col: df = df.rename(columns={lat_col: "lat"})
    if lon_col: df = df.rename(columns={lon_col: "lon"})

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])

    # site_id
    if "Sample Code" in df.columns:
        df["site_id"] = df["Sample Code"].astype(str)
    else:
        df["site_id"] = df.index.astype(str)

    return df.reset_index(drop=True)

@st.cache_data
def load_geojson(path):
    if not os.path.exists(path): return None
    try:
        return gpd.read_file(path)
    except:
        return None

# -------------------------------------------------------------
# POLLUTANT DETECTION
# -------------------------------------------------------------
def detect_pollutants(df):
    bases = set()
    months = ["Nov","Dec","Jan"]

    for col in df.columns:
        for m in months:
            if col.startswith(f"{m}_"):
                bases.add(col.split(f"{m}_", 1)[1])
            if col.endswith(f"_{m}"):
                bases.add(col.rsplit(f"_{m}", 1)[0])

    if not bases:
        numeric = df.select_dtypes(include=[np.number]).columns
        bases = {c for c in numeric if c not in ("lat","lon")}

    return sorted(bases)

def find_month_columns(df, pollutant):
    out = []
    for m in ["Nov","Dec","Jan"]:
        pref = f"{m}_{pollutant}"
        suff = f"{pollutant}_{m}"
        if pref in df.columns: out.append((pref,m))
        elif suff in df.columns: out.append((suff,m))
        else:
            for c in df.columns:
                if c.lower() == pref.lower() or c.lower() == suff.lower():
                    out.append((c,m))
                    break
    return out

def aggregate_pollutant(df, pollutant, months):
    pairs = find_month_columns(df, pollutant)
    cols = [c for c,m in pairs if m in months]
    if not cols: return pd.Series([0]*len(df), index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.max(axis=1).fillna(0)

# -------------------------------------------------------------
# RIVER MODELING FUNCTIONS
# -------------------------------------------------------------
def cluster_rivers(df):
    coords = df[["lat","lon"]].values
    return DBSCAN(eps=0.002, min_samples=2).fit(coords).labels_

def pca_sort(df, idxs):
    coords = df.loc[idxs, ["lat","lon"]].values
    pca = PCA(n_components=1)
    proj = pca.fit_transform(coords)
    order = np.argsort(proj[:,0])
    return np.array(idxs)[order]

def smooth_polyline(points):
    sm = []
    for i in range(len(points)):
        w1 = max(0, i-2)
        w2 = min(len(points), i+3)
        sm.append(points[w1:w2].mean(axis=0))
    return np.array(sm)

def draw_flow_arrows(map_obj, coords, color="#1f5cff"):
    for i in range(1, len(coords)):
        p1 = coords[i-1]
        p2 = coords[i]

        folium.PolyLine([tuple(p1),tuple(p2)],
                        color=color, weight=4, opacity=0.85).add_to(map_obj)

        folium.RegularPolygonMarker(
            location=tuple(p2),
            number_of_sides=3,
            radius=6,
            rotation=45,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(map_obj)

# -------------------------------------------------------------
# KRIGING ALONG RIVER
# -------------------------------------------------------------
def kriging_along_river(df, poll_vals, river_coords):
    t = np.zeros(len(river_coords))
    for i in range(1, len(river_coords)):
        lat1, lon1 = river_coords[i-1]
        lat2, lon2 = river_coords[i]
        t[i] = t[i-1] + np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2)

    site_t, site_vals = [], []
    for idx,row in df.iterrows():
        dists = (river_coords[:,0]-row["lat"])**2 + (river_coords[:,1]-row["lon"])**2
        closest = np.argmin(dists)
        site_t.append(t[closest])
        site_vals.append(poll_vals[idx])

    site_t = np.array(site_t).reshape(-1,1)
    site_vals = np.array(site_vals)

    kernel = 1.0*RBF(length_scale=0.02) + WhiteKernel(0.05)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gp.fit(site_t, site_vals)

    pred_mean, pred_std = gp.predict(t.reshape(-1,1), return_std=True)
    return pred_mean, pred_std

# -------------------------------------------------------------
# SOURCE DETECTION
# -------------------------------------------------------------
def detect_pollution_source(df, poll_series, sorted_idx):
    diffs = []
    for i in range(1, len(sorted_idx)):
        cur = sorted_idx[i]
        prev = sorted_idx[i-1]
        diffs.append((cur, poll_series[cur] - poll_series[prev]))

    if not diffs:
        return None, "Not enough points."

    cur, val = max(diffs, key=lambda x: x[1])
    if val <= 0:
        return None, "No positive spike."

    msg = f"Likely pollution source near **{df.loc[cur,'site_id']}** (increase: +{val:.2f})."
    return cur, msg

# -------------------------------------------------------------
# UI INPUTS
# -------------------------------------------------------------
df = load_data(DATA_PATH)
geo = load_geojson(GEOJSON_PATH)

pollutant_bases = detect_pollutants(df)
months_all = ["Nov","Dec","Jan"]

st.title("Punjab N-Choe Pollutant Visualizer — Ultra Edition")
st.markdown("### Kriging • Flow Arrows • Source Detection • DBSCAN Clustering")

st.sidebar.header("Controls")
pollutant = st.sidebar.selectbox("Pollutant", pollutant_bases)
months_selected = st.sidebar.multiselect("Months", months_all, default=months_all)

show_districts = st.sidebar.checkbox("Show District Boundaries", True)
cluster_markers = st.sidebar.checkbox("Cluster Markers", True)
show_kriging = st.sidebar.checkbox("Show Kriging Interpolation", True)
show_source = st.sidebar.checkbox("Detect Pollution Source", True)

search_query = st.sidebar.text_input("Search Site")

colormap = st.sidebar.selectbox("Colormap",
    ["RdBu","coolwarm","seismic","PiYG","PRGn"])

# -------------------------------------------------------------
# MAP RENDERING
# -------------------------------------------------------------
map_col, right_col = st.columns([2.6, 1])

with map_col:
    st.subheader("Map View")

    fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, control_scale=True)

    if geo is not None and show_districts:
        folium.GeoJson(
            geo,
            style_function=lambda f: {"color":"#444", "weight":1, "fillColor":"#00000000"}
        ).add_to(fmap)

    labels = cluster_rivers(df)
    uniq_clusters = sorted(set(labels))

    agg_vals = aggregate_pollutant(df, pollutant, months_selected)

    vmin, vmax = float(agg_vals.min()), float(agg_vals.max())
    if vmin == vmax: vmax = vmin + 1

    cmap = mpl.cm.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    def colorize(v):
        return mcolors.to_hex(sm.to_rgba(float(v)))

    marker_layer = MarkerCluster() if cluster_markers else fmap
    if cluster_markers: fmap.add_child(marker_layer)

    cluster_lines = {}
    cluster_orders = {}

    # Build synthetic rivers
    for c in uniq_clusters:
        idxs = np.where(labels == c)[0]
        if len(idxs) < 2:
            continue

        sorted_idx = pca_sort(df, idxs)
        cluster_orders[c] = sorted_idx

        coords = df.loc[sorted_idx, ["lat","lon"]].values
        smoothed = smooth_polyline(coords)
        cluster_lines[c] = smoothed

        draw_flow_arrows(fmap, smoothed)

    # Add markers
    df_plot = df.copy()
    if search_query.strip():
        df_plot = df[df["site_id"].str.contains(search_query, case=False, na=False)]

    for idx,row in df_plot.iterrows():
        val = agg_vals[idx]
        col = colorize(val)

        folium.CircleMarker(
            location=(row["lat"],row["lon"]),
            radius=7,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.9,
            tooltip=f"<b>{row['site_id']}</b><br>{pollutant}: {val:.2f}"
        ).add_to(marker_layer)

    # Kriging overlay
    if show_kriging:
        for c in cluster_lines:
            coords = cluster_lines[c]
            sorted_idx = cluster_orders[c]

            pred, _ = kriging_along_river(df, agg_vals, coords)

            for i in range(1, len(coords)):
                col = colorize(pred[i])
                folium.PolyLine(
                    [tuple(coords[i-1]), tuple(coords[i])],
                    color=col, weight=8, opacity=0.65
                ).add_to(fmap)

    # Pollution source detection
    source_info = "Detection disabled."
    source_site = None

    if show_source:
        for c in cluster_orders:
            sorted_idx = cluster_orders[c]
            site, info = detect_pollution_source(df, agg_vals, sorted_idx)
            if site is not None:
                source_site = site
                source_info = info

                folium.CircleMarker(
                    location=(df.loc[site,"lat"], df.loc[site,"lon"]),
                    radius=10,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.95,
                    tooltip=f"Likely Source: {df.loc[site,'site_id']}"
                ).add_to(fmap)

    map_data = st_folium(fmap, height=600, use_container_width=True,
                         returned_objects=["last_clicked"])

# -------------------------------------------------------------
# RIGHT PANEL — DETAILS
# -------------------------------------------------------------
with right_col:
    st.subheader("Selected Site")

    last_click = map_data.get("last_clicked")
    selected_idx = None

    if last_click:
        latc, lonc = last_click["lat"], last_click["lng"]
        dists = (df["lat"] - latc)**2 + (df["lon"] - lonc)**2
        if dists.min() < 0.0005:
            selected_idx = int(dists.idxmin())

    if selected_idx is None:
        st.info("Click a marker to view details.")
        st.stop()

    row = df.loc[selected_idx]
    st.markdown(f"### {row['site_id']}")
    st.write(f"Lat: {row['lat']}  \nLon: {row['lon']}")

    # Time-series for pollutant
    st.markdown("### Pollutant Over Time")
    pairs = find_month_columns(df, pollutant)

    if pairs:
        months = [m for _,m in pairs]
        vals = [row[c] for c,_ in pairs]

        fig = go.Figure()
        fig.add_bar(x=months, y=vals)
        fig.add_scatter(x=months, y=vals, mode="lines+markers")
        fig.update_layout(
            title=f"{pollutant} for {row['site_id']}",
            yaxis_title=pollutant,
            template="plotly_white",
            height=330
        )
        st.plotly_chart(fig, use_container_width=True)

    # Upstream / downstream
    st.markdown("### Upstream / Downstream")

    labels_here = cluster_rivers(df)
    c = labels_here[selected_idx]
    sorted_idx = cluster_orders.get(c, None)

    if sorted_idx is not None:
        pos = list(sorted_idx).index(selected_idx)
        upstream = sorted_idx[:pos]
        downstream = sorted_idx[pos+1:]

        if len(upstream) > 0:
            upstream_str = ", ".join(str(df.loc[u,"site_id"]) for u in upstream)
        else:
            upstream_str = "None"

        if len(downstream) > 0:
            downstream_str = ", ".join(str(df.loc[d,"site_id"]) for d in downstream)
        else:
            downstream_str = "None"

        st.write(f"Upstream: {upstream_str}")
        st.write(f"Downstream: {downstream_str}")

    # Pollution source
    st.markdown("### Pollution Source Detection")
    st.info(source_info)

# -------------------------------------------------------------
# LEGEND
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
### Legend
- **Blue arrows** → River flow direction  
- **Colored thick line** → Kriging interpolation  
- **Circle color** → Pollutant concentration  
- **Red circle** → Probable pollution source  
""")
st.markdown("---")

st.success("Ultra Edition Loaded Successfully — PCA + Kriging + Source Detection Active ✔")

