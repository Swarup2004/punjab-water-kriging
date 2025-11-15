# app.py
"""
Punjab River Pollutant Visualizer - patched + downstream differential + diverging colormaps + OSM tiles
Run:
    streamlit run app.py
"""

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
import plotly.express as px

import matplotlib as mpl
import matplotlib.colors as mcolors


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(layout="wide", page_title="Punjab River Pollutant Visualizer - Patched + Diff")

DATA_PATH = "./data/river_data_cleaned.csv"
GEOJSON_PATH = "assets/punjab_districts.geojson"

MAP_CENTER = [30.9, 75.8]
MAP_ZOOM = 7
PUNJAB_BBOX = [28.5, 73.5, 32.5, 77.5]


# -------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize month formats
    col_map = {}
    for c in df.columns:
        newc = (
            c.replace("Nov-24_", "Nov_")
             .replace("Dec-24_", "Dec_")
             .replace("Jan-25_", "Jan_")
             .replace("Nov-24", "Nov")
             .replace("Dec-24", "Dec")
             .replace("Jan-25", "Jan")
        )
        col_map[c] = newc
    df = df.rename(columns=col_map)

    # Normalize lat/lon columns
    lat_candidates = [c for c in df.columns if c.lower() in ("lat","latitude")]
    lon_candidates = [c for c in df.columns if c.lower() in ("lon","longitude")]

    if lat_candidates:
        df = df.rename(columns={lat_candidates[0]: "lat"})
    if lon_candidates:
        df = df.rename(columns={lon_candidates[0]: "lon"})

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)

    # Detect site id
    site_candidates = [c for c in df.columns if "site" in c.lower() or "location" in c.lower()]
    if site_candidates:
        df = df.rename(columns={site_candidates[0]: "site_id"})
    else:
        df["site_id"] = df.index.astype(str)

    return df


# -------------------------------------------------------
# POLLUTANT DETECTION
# -------------------------------------------------------
def detect_pollutants(df: pd.DataFrame) -> List[str]:
    months = ["Nov", "Dec", "Jan"]
    bases = set()
    for col in df.columns:
        for m in months:
            if col.startswith(m + "_"):
                bases.add(col.split(f"{m}_", 1)[1])
            if col.endswith("_" + m):
                bases.add(col.rsplit(f"_{m}", 1)[0])
    return sorted(list(bases))


def find_monthed_columns_for_pollutant(df, pollutant_base, months_list):
    cols = []
    for m in months_list:
        pref = f"{m}_{pollutant_base}"
        suff = f"{pollutant_base}_{m}"
        exact = None
        if pref in df.columns:
            exact = pref
        elif suff in df.columns:
            exact = suff
        else:
            for c in df.columns:
                if c.lower() == pref.lower():
                    exact = c
                    break
                if c.lower() == suff.lower():
                    exact = c
                    break
        if exact:
            cols.append((exact, m))
    return cols


def compute_aggregate_for_color(df, pollutant_base, months_selected):
    pairs = find_monthed_columns_for_pollutant(df, pollutant_base, months_selected)
    if not pairs:
        return pd.Series([0]*len(df), index=df.index)
    cols = [c for c, _ in pairs]
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.max(axis=1).fillna(0)


# -------------------------------------------------------
# DOWNSTREAM DIFFERENTIAL
# -------------------------------------------------------
def _principal_axis_sort_indices(lats, lons):
    coords = np.vstack([lats, lons]).T
    mask = ~np.isnan(coords).any(axis=1)
    if mask.sum() < 2:
        return np.arange(len(lats))

    valid = coords[mask] - coords[mask].mean(0)
    u, s, vh = np.linalg.svd(valid)
    pc = vh[0]
    proj = valid @ pc
    idxs = np.where(mask)[0]
    return idxs[np.argsort(proj)]


def compute_downstream_differential(df, series, signed=True):
    out = pd.Series(0.0, index=df.index)
    order = _principal_axis_sort_indices(df["lat"], df["lon"])

    vals = series[order].values
    diffs = np.zeros_like(vals)

    diffs[0] = vals[0]
    for i in range(1, len(vals)):
        if np.isnan(vals[i]) or np.isnan(vals[i-1]):
            diffs[i] = 0
        else:
            diffs[i] = vals[i] - vals[i-1]

    if not signed:
        diffs = np.clip(diffs, 0, None)

    out[order] = diffs
    return out


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Data file missing: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

gdf = None
if os.path.exists(GEOJSON_PATH):
    try:
        gdf = gpd.read_file(GEOJSON_PATH)
    except:
        gdf = None

pollutant_bases = detect_pollutants(df)


# -------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------
st.title("Punjab River Pollutant Visualizer — Patched + Downstream Differential")
st.markdown("Click markers to inspect. Choose pollutant, months, and differential mode.")

st.sidebar.header("Controls")

pollutant = st.sidebar.selectbox("Pollutant", pollutant_bases)
months_all = ["Nov", "Dec", "Jan"]
months_selected = st.sidebar.multiselect("Months", months_all, default=months_all)

marker_size_option = st.sidebar.selectbox("Marker size by", ["fixed", "value"], index=1)
radius_fixed = st.sidebar.slider("Fixed radius", 4, 20, 8)
cluster_toggle = st.sidebar.checkbox("Cluster markers", True)
search_text = st.sidebar.text_input("Search site")
show_districts = st.sidebar.checkbox("Show district boundaries", True)

diverging_cmaps = ["RdBu", "coolwarm", "PiYG", "PRGn", "PuOr", "RdYlBu", "Spectral", "seismic"]
cmap_choice = st.sidebar.selectbox("Colormap", diverging_cmaps)

use_diff = st.sidebar.checkbox("Use downstream differential", False)
diff_mode = st.sidebar.selectbox("Differential type", ["signed", "positive-only"])


# -------------------------------------------------------
# LAYOUT
# -------------------------------------------------------
map_col, info_col = st.columns([2.3, 1])

# -------------------------------------------------------
# MAIN MAP
# -------------------------------------------------------
with map_col:
    st.subheader("Map")

    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        height=600,
        width="100%",
        tiles=None
    )

    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB Positron").add_to(m)
    folium.LayerControl().add_to(m)

    if gdf is not None and show_districts:
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda x: {"fillColor":"#00000000","color":"#555","weight":1}
        ).add_to(m)

    agg = compute_aggregate_for_color(df, pollutant, months_selected)

    if use_diff:
        signed = diff_mode == "signed"
        color_vals = compute_downstream_differential(df, agg, signed=signed)
    else:
        color_vals = agg

    vmin, vmax = float(color_vals.min()), float(color_vals.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1

    cmap = mpl.cm.get_cmap(cmap_choice)
    norm = (
        mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        if use_diff else mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if cluster_toggle:
        cl = MarkerCluster().add_to(m)
    else:
        cl = m

    filtered = df if not search_text else df[df["site_id"].astype(str).str.contains(search_text, case=False)]

    for idx, row in filtered.iterrows():
        lat, lon = row["lat"], row["lon"]
        val = color_vals.loc[idx]
        color = mcolors.to_hex(sm.to_rgba(float(val)))

        if marker_size_option == "value":
            r = np.interp(val, [vmin, vmax], [5, 18])
        else:
            r = radius_fixed

        tooltip = f"<b>{row['site_id']}</b><br>({lat:.5f}, {lon:.5f})"
        folium.CircleMarker(
            [lat, lon],
            radius=r,
            color=color,
            fill=True,
            fill_color=color,
            tooltip=tooltip
        ).add_to(cl)

    # LEGEND (simple)
    legend = f"""
    <div style="position: fixed; bottom: 40px; left: 12px; z-index: 9999;
                background-color: white; padding: 8px; border-radius: 6px;">
        <b>Pollutant:</b> {pollutant}<br>
        Mode: {"Differential" if use_diff else "Aggregate"}<br>
        Range: {vmin:.2f} → {vmax:.2f}<br>
        Colormap: {cmap_choice}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    map_data = st_folium(
        m,
        key="mainmap",
        height=600,
        use_container_width=True,
        returned_objects=["last_clicked"]
    )

    last_click = map_data.get("last_clicked")


# -------------------------------------------------------
# SITE DETAILS
# -------------------------------------------------------
with info_col:
    st.subheader("Selected Site")

    site_ids = df["site_id"].astype(str).tolist()
    site_choice = st.selectbox("Choose site", ["(none)"] + site_ids)

    selected_idx = None
    if site_choice != "(none)":
        selected_idx = df.index[df["site_id"].astype(str)==site_choice][0]
    elif last_click:
        latc, lonc = last_click["lat"], last_click["lng"]
        dists = (df["lat"] - latc)**2 + (df["lon"] - lonc)**2
        if dists.min() < 0.0001:
            selected_idx = int(dists.idxmin())

    if selected_idx is None:
        st.info("Click a site or choose from dropdown.")
        st.stop()

    row = df.loc[selected_idx]
    st.markdown(f"### {row['site_id']} (index {selected_idx})")
    st.write(f"Coordinates: {row['lat']:.6f}, {row['lon']:.6f}")

    with st.expander("Zoom map"):
        sm = folium.Map(location=[row["lat"], row["lon"]], zoom_start=14)
        folium.CircleMarker([row["lat"], row["lon"]], radius=8, color="red", fill=True).add_to(sm)
        st_folium(sm, height=250, width=350)

    pollutant_series = {}
    for pb in pollutant_bases:
        pairs = find_monthed_columns_for_pollutant(df, pb, ["Nov","Dec","Jan"])
        if pairs:
            months = [m for _, m in pairs]
            cols = [c for c, _ in pairs]
            vals = [row.get(c, np.nan) for c in cols]
            pollutant_series[pb] = {"months":months, "cols":cols, "values":vals}

    st.markdown("### Pollutant charts")

    choose = st.multiselect("Choose pollutants", pollutant_series.keys(),
                            default=list(pollutant_series.keys())[:2])

    for pb in choose:
        months = pollutant_series[pb]["months"]
        vals = pollutant_series[pb]["values"]

        fig = go.Figure()
        fig.add_bar(x=months, y=vals)
        fig.add_scatter(x=months, y=vals, mode="lines+markers")
        fig.update_layout(title=f"{pb}", height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    rows = []
    for pb, obj in pollutant_series.items():
        for col, mm, v in zip(obj["cols"], obj["months"], obj["values"]):
            rows.append({"metric": f"{pb}_{mm}", "value": v})
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df.astype(str), height=300)

    # Export
    site_only = df[df["site_id"].astype(str)==str(row["site_id"])]
    st.download_button("Download site CSV", site_only.to_csv(index=False),
                       file_name=f"{row['site_id']}.csv")
