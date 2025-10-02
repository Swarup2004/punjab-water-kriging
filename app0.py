"""
Streamlit app: Punjab river pollutant visualizer
Requirements (pip):
  pip install streamlit pandas geopandas folium streamlit-folium plotly pyproj scikit-learn

Optional (for export to PNG):
  pip install kaleido

Run locally:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from typing import List, Dict
from functools import lru_cache
import os

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/river_data_cleaned.csv"         # change if needed
GEOJSON_PATH = "assets/punjab_districts.geojson" # change if needed
TITLE = "Punjab River Pollutant Visualizer"
DEFAULT_POLLUTANT = None  # auto choose highest-range pollutant
MAP_CENTER = [30.9, 75.8]  # center roughly Punjab
MAP_ZOOM = 7
# bounding box to restrict to Punjab (approx)
PUNJAB_BBOX = [28.5, 73.5, 32.5, 77.5]  # [south, west, north, east]
# ----------------------------

st.set_page_config(layout="wide", page_title=TITLE)

st.title(TITLE)
st.markdown(
    """
    Visualize water pollutant readings across Punjab.  
    Click a marker to see interactive month-to-month charts and export data.
    """
)

# ----------------------------
# Helpers & caching
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # unify column names
    df.columns = [c.strip() for c in df.columns]
    # try to coerce lat/lon
    lat_cols = [c for c in df.columns if c.lower() in ("lat","latitude")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon","longitude","long")]
    if lat_cols and lon_cols:
        df = df.rename(columns={lat_cols[0]: "lat", lon_cols[0]: "lon"})
    # ensure lat/lon numeric and drop rows without coords
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    # try to ensure site id/name
    if "site_id" not in df.columns and "site" in df.columns:
        df = df.rename(columns={"site":"site_id"})
    if "site_id" not in df.columns:
        # make a fallback site id
        df["site_id"] = df.index.astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    gdf = gpd.read_file(path)
    return gdf

def detect_pollutants(df: pd.DataFrame) -> List[str]:
    """
    Detect pollutant base names by finding patterns like <pollutant>_Nov/Dec/Jan.
    Returns list of base pollutant names.
    """
    months = ["Nov","Dec","Jan"]
    pollutant_bases = set()
    for col in df.columns:
        for m in months:
            if col.endswith(f"_{m}") or col.endswith(f"-{m}") or col.endswith(f" {m}"):
                base = col.rsplit("_",1)[0].rsplit("-",1)[0].rsplit(" ",1)[0]
                pollutant_bases.add(base)
    # fallback: if columns like pH_Nov present then good; else attempt to detect columns with Nov/Dec/Jan in header anywhere
    if not pollutant_bases:
        for col in df.columns:
            for m in months:
                if m.lower() in col.lower():
                    base = col.replace(m, "").replace("_","").replace(".","").strip("_ -")
                    pollutant_bases.add(base)
    return sorted(pollutant_bases)

def build_site_summary(row, pollutant_bases, months):
    # produce a small summary string for tooltip
    parts = []
    for pb in pollutant_bases:
        vals = []
        for m in months:
            c = f"{pb}_{m}"
            if c in row:
                v = row[c]
                if pd.notna(v):
                    vals.append(f"{m}:{v}")
        if vals:
            parts.append(f"{pb}({' / '.join(vals)})")
    return "<br>".join(parts)

def compute_aggregate_for_color(df, pollutant_base, months_selected):
    """
    Compute an aggregate value used for marker color/size.
    Returns a Pandas Series aligned with df.
    """
    cols = [f"{pollutant_base}_{m}" for m in months_selected if f"{pollutant_base}_{m}" in df.columns]
    if not cols:
        return pd.Series([0.0] * len(df), index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    agg = vals.max(axis=1, skipna=True).fillna(0)
    return agg


def percent_change(a, b):
    try:
        a = float(a); b = float(b)
        if a == 0:
            return np.nan
        return (b - a) / abs(a) * 100.0
    except Exception:
        return np.nan

# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at `{DATA_PATH}`. Please place `river_data_cleaned.csv` at this path or modify DATA_PATH.")
    st.stop()

if not os.path.exists(GEOJSON_PATH):
    st.warning(f"GeoJSON file not found at `{GEOJSON_PATH}`. District boundary layer will be skipped.")
    gdf = None
else:
    gdf = load_geojson(GEOJSON_PATH)

df = load_data(DATA_PATH)

# detect pollutant base names
months_all = ["Nov","Dec","Jan"]
pollutant_bases = detect_pollutants(df)
if not pollutant_bases:
    st.warning("Could not detect pollutant columns with Nov/Dec/Jan patterns. Make sure your CSV has columns like 'pH_Nov', 'BOD_Dec', etc. App will still try to display numeric columns.")
    # fallback: choose numeric columns excluding lat/lon
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("lat","lon")]
    pollutant_bases = numeric_cols[:5]  # best-effort
# default pollutant
if DEFAULT_POLLUTANT is None:
    selected_pollutant = pollutant_bases[0]
else:
    selected_pollutant = DEFAULT_POLLUTANT

# ----------------------------
# Sidebar (controls)
# ----------------------------
st.sidebar.header("Controls")
pollutant = st.sidebar.selectbox("Marker encoding pollutant", pollutant_bases, index=0)
months_selected = st.sidebar.multiselect("Months to show in tooltips", months_all, default=months_all)
marker_size_option = st.sidebar.selectbox("Marker size by", ["fixed", "value"], index=1)
fixed_marker_px = st.sidebar.slider("Fixed marker radius (px)", 4, 20, 8)
cluster_toggle = st.sidebar.checkbox("Use marker clustering", value=True)
search_text = st.sidebar.text_input("Search site by name/ID (partial)", value="")
show_districts = st.sidebar.checkbox("Show district boundaries", value=True)
export_png = st.sidebar.checkbox("Allow PNG export of charts (kaleido required)", value=True)

# ----------------------------
# Top row: map + controls column
# ----------------------------
map_col, right_col = st.columns([2.2, 1])

with map_col:
    st.subheader("Map")
    # Build folium map
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, min_zoom=6, max_bounds=True)
    # restrict bounding box
    m.fit_bounds([[PUNJAB_BBOX[0], PUNJAB_BBOX[1]], [PUNJAB_BBOX[2], PUNJAB_BBOX[3]]])

    # Add district boundaries
    if gdf is not None and show_districts:
        folium.GeoJson(
            gdf.to_json(),
            name="Districts",
            style_function=lambda feat: {
                "fillColor": "#ffffff00",
                "color": "#444444",
                "weight": 1,
                "opacity": 0.6
            }
        ).add_to(m)

    # compute color aggregate
    agg = compute_aggregate_for_color(df, pollutant, months_selected)
    # normalize for color mapping
    val_min, val_max = float(np.nanmin(agg)), float(np.nanmax(agg))
    if val_max == val_min:
        val_max = val_min + 1.0

    # color scale function
    def value_to_color(v):
        # simple blue->red scale
        try:
            frac = (v - val_min) / (val_max - val_min)
            frac = min(max(frac, 0.0), 1.0)
            return px.colors.label_rgb(px.colors.sample_colorscale("Viridis", [frac])[0])
        except Exception:
            return "gray"

    # add markers
    marker_cluster = MarkerCluster() if cluster_toggle else None
    if marker_cluster:
        m.add_child(marker_cluster)

    # iterate rows, apply search filter
    filtered_df = df.copy()
    if search_text.strip():
        filtered_df = filtered_df[filtered_df.apply(lambda r: search_text.lower() in str(r.get("site_id","")).lower(), axis=1)]

    for idx, row in filtered_df.iterrows():
        lat = row["lat"]; lon = row["lon"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        val = agg.loc[idx] if idx in agg.index else 0.0
        color = value_to_color(val)
        if marker_size_option == "value":
            # scale radius between 4..18
            try:
                radius = float(np.interp(val, [val_min, val_max], [4, 18]))
            except Exception:
                radius = fixed_marker_px
        else:
            radius = fixed_marker_px
        # tooltip content (basic)
        tooltip_html = f"<b>{row.get('site_id','')}</b><br>({lat:.5f}, {lon:.5f})<br>"
        tooltip_html += build_site_summary(row, pollutant_bases, months_selected)
        # quality flags if present
        qflags = [c for c in df.columns if "flag" in c.lower() or "quality" in c.lower()]
        if qflags:
            qvals = []
            for q in qflags:
                if q in row and pd.notna(row[q]):
                    qvals.append(f"{q}:{row[q]}")
            if qvals:
                tooltip_html += "<br>" + "<br>".join(qvals)

        popup_id = f"site-{idx}"
        # add marker
        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=folium.Tooltip(tooltip_html, sticky=True, direction="top"),
        )
        if marker_cluster:
            marker_cluster.add_child(marker)
        else:
            marker.add_to(m)

        # create a link in popup to trigger streamlit side panel via URL fragment hack: we will show details by clicking the marker and then the user clicks link to "Show details"
        popup_html = f"""
        <div>
          <b>{row.get('site_id','')}</b><br/>
          Coordinates: {lat:.5f}, {lon:.5f}<br/>
          <a href="#" onclick="window.parent.postMessage({{'type':'site_clicked','idx': {int(idx)}}}, '*')">Show details</a>
        </div>
        """
        folium.Popup(popup_html, max_width=300).add_to(marker)

    # add legend (simple)
    legend_html = f"""
     <div style="position: fixed; bottom: 45px; left: 10px; z-index:9999; background-color: white; padding: 8px; border-radius:6px;">
       <b>Legend</b><br>
       Pollutant: <b>{pollutant}</b><br>
       Value range: {val_min:.2f} — {val_max:.2f}<br>
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # embed map
    map_data = st_folium(m, width="100%", height=600, returned_objects=["last_clicked", "all_popups"])

    # Listen for postMessage (the JS postMessage triggered by popup link). streamlit_folium returns last_clicked - but not our custom posted message.
    # However st_folium returns "last_object_clicked" for click; we use that to select site.
    last_click = map_data.get("last_clicked")
    # We also try `last_object_clicked` or popups: streamlit_folium may not return our postMessage.
    # As fallback, allow user to click marker and then choose from a dropdown on right panel.

with right_col:
    st.subheader("Selected site")
    # Selection UI in right column
    site_idx_selection = st.number_input("Selected site index (or leave -1)", min_value=-1, max_value=len(df)-1, value=-1, step=1)
    # prefer last_click detection
    selected_idx = None
    if last_click and isinstance(last_click, dict):
        # last_click returns {"lat":..., "lng":...}
        lat_lng = (last_click.get("lat"), last_click.get("lng"))
        # find nearest site within small tolerance
        if lat_lng[0] is not None:
            dists = ((df["lat"] - lat_lng[0])**2 + (df["lon"] - lat_lng[1])**2)
            selected_idx = int(dists.idxmin())
            # if too far, ignore
            if dists.min() > 0.0008:  # rough threshold (~1km)
                selected_idx = None

    if site_idx_selection >= 0:
        selected_idx = int(site_idx_selection)

    if selected_idx is None or selected_idx < 0:
        st.info("Click a marker on the map or enter a site index to view details.")
    else:
        # prepare and show details for selected site
        row = df.loc[selected_idx]
        st.markdown(f"### {row.get('site_id','Site')}  (index: {selected_idx})")
        st.write(f"Coordinates: **{row['lat']:.6f}**, **{row['lon']:.6f}**")

        # small map zoomed to point
        with st.expander("Small map / zoom"):
            sm = folium.Map(location=[row['lat'], row['lon']], zoom_start=13)
            folium.CircleMarker(location=[row['lat'], row['lon']], radius=8, color="red", fill=True).add_to(sm)
            st_folium(sm, width=400, height=300)

        # build pollutant charts
        # Gather pollutant time series for this row
        available_pollutants = []
        pollutant_series = {}
        for pb in pollutant_bases:
            cols = [f"{pb}_{m}" for m in months_all if f"{pb}_{m}" in df.columns]
            if cols:
                vals = [row[c] if c in row else np.nan for c in cols]
                pollutant_series[pb] = dict(months=[c.rsplit("_",1)[1] for c in cols], values=vals, cols=cols)
                available_pollutants.append(pb)

        if not available_pollutants:
            st.warning("No pollutant month-columns found for this site. Show raw numeric columns instead.")
            st.dataframe(row.head(30))
        else:
            # Multi-select pollutants to plot
            chosen = st.multiselect("Chart pollutants", available_pollutants, default=available_pollutants[:2])
            for pb in chosen:
                data = pollutant_series[pb]
                months = data["months"]
                vals = [None if pd.isna(x) else float(x) for x in data["values"]]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=months, y=vals, name=f"{pb} (value)"))
                # add line for trend
                fig.add_trace(go.Scatter(x=months, y=vals, mode="lines+markers", name=f"{pb} trend"))
                fig.update_layout(title=f"{pb} — monthly values", yaxis_title=f"{pb}", template="plotly_white", height=350)
                st.plotly_chart(fig, use_container_width=True)

                # percent change Nov -> Jan
                if len(vals) >= 2:
                    # attempt Nov -> Jan if present
                    change_text = ""
                    try:
                        # find first and last month vals that are not None
                        pairs = [(i,v) for i,v in enumerate(vals) if v is not None]
                        if len(pairs) >= 2:
                            a_idx, a_val = pairs[0]
                            b_idx, b_val = pairs[-1]
                            pc = percent_change(a_val, b_val)
                            if not np.isnan(pc):
                                change_text = f"{pb}: {pc:.1f}% change from {months[a_idx]} → {months[b_idx]}."
                            else:
                                change_text = f"{pb}: percent change not defined (initial value 0 or missing)."
                        else:
                            change_text = f"{pb}: not enough data to compute percent change."
                    except Exception as e:
                        change_text = f"Error computing percent change: {e}"
                    st.write(change_text)

            # show small summary table
            st.markdown("**Numeric summary**")
            summary_rows = {}
            for pb, obj in pollutant_series.items():
                for col, m, v in zip(obj["cols"], obj["months"], obj["values"]):
                    summary_rows[f"{pb}_{m}"] = v
            summary_df = pd.DataFrame(summary_rows.items(), columns=["metric","value"])
            st.dataframe(summary_df)

            # Simple interpretation rules if thresholds present in CSV or known thresholds
            # Example: if CSV has columns like "<pollutant>_threshold" or we define simple safe thresholds:
            safe_thresholds = {
                # sample thresholds — adjust per domain knowledge or CSV columns
                "Nitrate": 50,     # mg/L
                "BOD": 3,          # mg/L
                "DO": 5,           # mg/L (minimum acceptable)
                "pH": (6.5, 8.5),  # range
            }
            interpretation = []
            for pb, obj in pollutant_series.items():
                vals = [v for v in obj["values"] if pd.notna(v)]
                if not vals:
                    continue
                mean_val = np.mean(vals)
                thr = safe_thresholds.get(pb)
                if thr:
                    if isinstance(thr, tuple):
                        lo, hi = thr
                        if (mean_val < lo) or (mean_val > hi):
                            interpretation.append(f"{pb}: mean {mean_val:.2f} outside safe range {lo}-{hi}.")
                        else:
                            interpretation.append(f"{pb}: mean {mean_val:.2f} within safe range.")
                    else:
                        if mean_val > thr:
                            interpretation.append(f"{pb}: mean {mean_val:.2f} exceeds threshold {thr}.")
                        else:
                            interpretation.append(f"{pb}: mean {mean_val:.2f} within threshold {thr}.")
            if interpretation:
                st.markdown("**Automated interpretation**")
                for t in interpretation:
                    st.write("- " + t)

        # export options
        st.markdown("---")
        st.markdown("**Exports**")
        # CSV export: offer the single row or all rows for that site (if multiple entries exist)
        site_rows = df[df.get("site_id", "") == row.get("site_id")]
        if site_rows.empty:
            site_rows = pd.DataFrame([row])
        csv_bytes = site_rows.to_csv(index=False).encode("utf-8")
        st.download_button("Download site CSV", data=csv_bytes, file_name=f"{row.get('site_id')}_data.csv", mime="text/csv")

        # PNG export of charts: attempt to save the last plotted chart(s)
        # We'll produce a combined figure (if chosen pollutants exist)
        if chosen:
            fig_all = go.Figure()
            for pb in chosen:
                obj = pollutant_series[pb]
                months = obj["months"]
                vals = [None if pd.isna(x) else float(x) for x in obj["values"]]
                fig_all.add_trace(go.Bar(x=months, y=vals, name=pb))
            fig_all.update_layout(title=f"{row.get('site_id')} — selected pollutants", template="plotly_white", height=450)
            try:
                # try to export to PNG via kaleido
                img_bytes = fig_all.to_image(format="png", scale=2)
                st.download_button("Download chart PNG", data=img_bytes, file_name=f"{row.get('site_id')}_chart.png", mime="image/png")
            except Exception as e:
                st.warning("PNG export unavailable (kaleido may not be installed). You can still download the CSV.")
                st.write(f"PNG export error: {e}")

        # link to raw CSV row(s)
        st.markdown("**Raw CSV rows**")
        st.dataframe(site_rows)

# ----------------------------
# Footer / instructions / test
# ----------------------------
st.markdown("---")
st.markdown(
    """
    **Notes & tips**
    - Click a marker, then in the right panel enter the site index printed in the console or click the marker and use the small link to 'Show details'.  
    - To adapt to your CSV, ensure columns are named like `BOD_Nov`, `BOD_Dec`, `BOD_Jan` (same pollutant base name, month suffix).
    - If you have many points, clustering is enabled by default.
    """
)

# small test/run button
def _run_testsample():
    st.write("Basic data sample")
    st.dataframe(df.head(5))
if st.button("Show data sample"):
    _run_testsample()

