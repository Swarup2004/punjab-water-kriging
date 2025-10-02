# app.py
"""
Punjab River Pollutant Visualizer - patched version
Run:
    streamlit run app.py

Ensure DATA_PATH points to your CSV (example uses the uploaded '/mnt/data/ilgc_data - Sheet1 (1).csv').
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

st.set_page_config(layout="wide", page_title="Punjab River Pollutant Visualizer - Patched")

# ---------- CONFIG ----------
DATA_PATH = "./data/river_data_cleaned.csv"  # update if needed
GEOJSON_PATH = "assets/punjab_districts.geojson"
MAP_CENTER = [30.9, 75.8]
MAP_ZOOM = 7
PUNJAB_BBOX = [28.5, 73.5, 32.5, 77.5]
# ----------------------------

# ----------------------------
# CACHED I/O & HELPERS
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and normalize column names: convert Nov-24_ to Nov_, unify lat/lon and site columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise some month-year patterns often present in your file -> Nov_, Dec_, Jan_
    col_map = {}
    for c in df.columns:
        newc = c
        newc = newc.replace("Nov-24_", "Nov_").replace("Dec-24_", "Dec_").replace("Jan-25_", "Jan_")
        newc = newc.replace("Nov-24 ", "Nov_").replace("Dec-24 ", "Dec_").replace("Jan-25 ", "Jan_")
        newc = newc.replace("Nov-24", "Nov").replace("Dec-24", "Dec").replace("Jan-25", "Jan")
        col_map[c] = newc
    df = df.rename(columns=col_map)

    # lat/lon detection and rename
    lat_candidates = [c for c in df.columns if c.lower() in ("lat", "latitude", "latitude (deg)")]
    lon_candidates = [c for c in df.columns if c.lower() in ("lon", "longitude", "long", "longitude (deg)")]
    if lat_candidates:
        df = df.rename(columns={lat_candidates[0]: "lat"})
    if lon_candidates:
        df = df.rename(columns={lon_candidates[0]: "lon"})

    # site id detection
    site_candidates = [c for c in df.columns if c.lower() in ("site_id","site","sample location","sample code","sample_location","location")]
    if site_candidates:
        df = df.rename(columns={site_candidates[0]: "site_id"})
    else:
        # try other heuristics
        other = [c for c in df.columns if "site" in c.lower() or "sample" in c.lower() or "location" in c.lower()]
        if other:
            df = df.rename(columns={other[0]: "site_id"})
        else:
            df["site_id"] = df.index.astype(str)

    # coerce lat/lon to numeric and drop rows without coordinates
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)
    else:
        # leave as-is; map will warn later
        pass

    return df

def detect_pollutants(df: pd.DataFrame) -> List[str]:
    """
    Detect pollutant base names for columns like:
      - Nov_pH (prefix)
      - pH_Nov (suffix)
    Returns unique pollutant base names (strings).
    """
    months = ["Nov", "Dec", "Jan"]
    pollutant_bases = set()
    for col in df.columns:
        for m in months:
            if col.startswith(f"{m}_"):
                pollutant_bases.add(col.split(f"{m}_", 1)[1])
            if col.endswith(f"_{m}"):
                pollutant_bases.add(col.rsplit(f"_{m}", 1)[0])
    # fallback: choose numeric columns excluding lat/lon/site
    if not pollutant_bases:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pollutant_bases = set([c for c in numeric_cols if c not in ("lat","lon")][:12])
    return sorted(pollutant_bases)

def find_monthed_columns_for_pollutant(df: pd.DataFrame, pollutant_base: str, months_list: List[str]):
    """
    For a pollutant base (e.g., 'pH' or 'TotalBOD'), find columns matching
    either Nov_pH, pH_Nov, etc. Returns list of (colname, month_label) in months order.
    """
    cols = []
    for m in months_list:
        # prefer prefix style m_pollutant (Nov_pH)
        pref = f"{m}_{pollutant_base}"
        suff = f"{pollutant_base}_{m}"
        if pref in df.columns:
            cols.append((pref, m))
        elif suff in df.columns:
            cols.append((suff, m))
        else:
            # try case-insensitive variants
            for c in df.columns:
                if c.lower() == pref.lower():
                    cols.append((c, m)); break
                if c.lower() == suff.lower():
                    cols.append((c, m)); break
            else:
                # column missing for this month; skip
                pass
    return cols  # e.g. [('Nov_pH','Nov'), ('Dec_pH','Dec'), ('Jan_pH','Jan')]

def compute_aggregate_for_color(df: pd.DataFrame, pollutant_base: str, months_selected: List[str]) -> pd.Series:
    """
    Return pd.Series aligned with df.index representing an aggregate (max across selected months).
    """
    col_month_pairs = find_monthed_columns_for_pollutant(df, pollutant_base, months_selected)
    cols = [p for p, m in col_month_pairs]
    if not cols:
        # return zeros series aligned to df
        return pd.Series([0.0] * len(df), index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    agg = vals.max(axis=1, skipna=True).fillna(0)
    return pd.Series(agg.values, index=df.index)

def percent_change(a, b):
    try:
        a = float(a); b = float(b)
        if a == 0:
            return np.nan
        return (b - a) / abs(a) * 100.0
    except Exception:
        return np.nan

# ----------------------------
# Load files
# ----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at `{DATA_PATH}`. Update DATA_PATH variable in the script.")
    st.stop()

df = load_data(DATA_PATH)

gdf = None
if os.path.exists(GEOJSON_PATH):
    try:
        gdf = gpd.read_file(GEOJSON_PATH)
    except Exception:
        gdf = None

# Detect pollutants
pollutant_bases = detect_pollutants(df)

# ----------------------------
# UI controls
# ----------------------------
st.title("Punjab River Pollutant Visualizer — Patched")
st.markdown("Click a marker to select a site. Use the controls to change pollutant encoding and months.")

st.sidebar.header("Controls")
pollutant = st.sidebar.selectbox("Marker encoding pollutant", pollutant_bases if pollutant_bases else ["(none)"])
months_all = ["Nov", "Dec", "Jan"]
months_selected = st.sidebar.multiselect("Months used for tooltip/aggregates", months_all, default=months_all)
marker_size_option = st.sidebar.selectbox("Marker size by", ["fixed", "value"], index=1)
fixed_marker_px = st.sidebar.slider("Fixed marker radius (px)", 4, 20, 8)
cluster_toggle = st.sidebar.checkbox("Use marker clustering", value=True)
search_text = st.sidebar.text_input("Search site by name/ID (partial)", value="")
show_districts = st.sidebar.checkbox("Show district boundaries", value=True)
allow_png = st.sidebar.checkbox("Allow PNG export (kaleido)", value=True)

# ----------------------------
# Layout: map + details
# ----------------------------
map_col, details_col = st.columns([2.2, 1])

with map_col:
    st.subheader("Map")
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, control_scale=True)
    m.fit_bounds([[PUNJAB_BBOX[0], PUNJAB_BBOX[1]], [PUNJAB_BBOX[2], PUNJAB_BBOX[3]]])

    # district overlay
    if gdf is not None and show_districts:
        folium.GeoJson(gdf.to_json(), name="districts",
                       style_function=lambda feat: {"fillColor": "#ffffff00", "color": "#333333", "weight": 1, "opacity": 0.6}).add_to(m)

    # compute aggregate series for chosen pollutant (safe fallback)
    if pollutant and pollutant != "(none)":
        agg_series = compute_aggregate_for_color(df, pollutant, months_selected)
    else:
        agg_series = pd.Series([0.0]*len(df), index=df.index)

    val_min = float(np.nanmin(agg_series)) if len(agg_series) > 0 else 0.0
    val_max = float(np.nanmax(agg_series)) if len(agg_series) > 0 else (val_min + 1.0)
    if val_min == val_max:
        val_max = val_min + 1.0

    def value_to_color(v):
        try:
            frac = (v - val_min) / (val_max - val_min)
            frac = min(max(frac, 0.0), 1.0)
            c = px.colors.sample_colorscale("Viridis", [frac])[0]  # e.g. "rgb(...)"
            return c
        except Exception:
            return "gray"

    # cluster
    marker_cluster = MarkerCluster() if cluster_toggle else None
    if marker_cluster:
        m.add_child(marker_cluster)

    # filtered DF for search
    if search_text.strip():
        filtered_df = df[df["site_id"].astype(str).str.contains(search_text, case=False, na=False)]
    else:
        filtered_df = df

    # add markers
    for idx, row in filtered_df.iterrows():
        lat = row.get("lat"); lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        val = float(agg_series.loc[idx]) if idx in agg_series.index else 0.0
        color = value_to_color(val)
        radius = float(np.interp(val, [val_min, val_max], [4, 18])) if marker_size_option == "value" else fixed_marker_px

        # tooltip: include pollutant snippets for selected months
        t_lines = [f"<b>{row.get('site_id','')}</b>", f"({lat:.5f}, {lon:.5f})"]
        for pb in pollutant_bases:
            pairs = find_monthed_columns_for_pollutant(df, pb, months_selected)
            parts = []
            for col, mm in pairs:
                v = row.get(col)
                if pd.notna(v):
                    parts.append(f"{mm}:{v}")
            if parts:
                t_lines.append(f"{pb}: {' / '.join(parts)}")
        tooltip_html = "<br>".join(t_lines)

        marker = folium.CircleMarker(location=[lat, lon], radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.8,
                                     tooltip=folium.Tooltip(tooltip_html, sticky=True))
        popup_html = f"<div><b>{row.get('site_id','')}</b><br/>Index: {idx}<br/>Coords: {lat:.5f}, {lon:.5f}</div>"
        folium.Popup(popup_html, max_width=300).add_to(marker)

        if marker_cluster:
            marker_cluster.add_child(marker)
        else:
            marker.add_to(m)

    # legend
    legend_html = f"""
    <div style="position: fixed; bottom: 45px; left: 10px; z-index:9999; background-color: rgba(255,255,255,0.9); padding:8px; border-radius:6px;">
      <b>Legend</b><br>
      Pollutant: <b>{pollutant}</b><br>
      Range: {val_min:.2f} — {val_max:.2f}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width="100%", height=600, returned_objects=["last_clicked"])
    last_click = map_data.get("last_clicked") if isinstance(map_data, dict) else None

with details_col:
    st.subheader("Selected site")
    # deterministic dropdown + numeric input + map click fallback
    site_list = df["site_id"].astype(str).tolist()
    site_dropdown = st.selectbox("Choose a site", options=["(none)"] + site_list, index=0)
    idx_input = st.number_input("Or enter site index (or -1)", min_value=-1, max_value=len(df)-1, value=-1, step=1)

    selected_idx = None
    if idx_input >= 0:
        selected_idx = int(idx_input)
    elif site_dropdown != "(none)":
        matches = df.index[df["site_id"].astype(str) == site_dropdown].tolist()
        selected_idx = matches[0] if matches else None
    elif last_click and last_click.get("lat") is not None:
        lat_click, lon_click = last_click.get("lat"), last_click.get("lng")
        dists = ((df["lat"] - lat_click)**2 + (df["lon"] - lon_click)**2)
        closest = int(dists.idxmin())
        if dists.min() <= 0.0009:
            selected_idx = closest

    if selected_idx is None:
        st.info("Select a site with the dropdown, index, or click a marker on the map.")
    else:
        if selected_idx not in df.index:
            st.error("Index not found in data.")
        else:
            row = df.loc[selected_idx]
            st.markdown(f"### {row.get('site_id','Site')} (index: {selected_idx})")
            st.write(f"Coordinates: **{row['lat']:.6f}**, **{row['lon']:.6f}**")

            with st.expander("Small map / zoom"):
                sm = folium.Map(location=[row['lat'], row['lon']], zoom_start=13)
                folium.CircleMarker(location=[row['lat'], row['lon']], radius=8, color="red", fill=True).add_to(sm)
                st_folium(sm, width=350, height=250)

            # Build pollutant time-series per pollutant for this site (handles prefix & suffix)
            pollutant_series = {}
            for pb in pollutant_bases:
                col_month_pairs = find_monthed_columns_for_pollutant(df, pb, months_all)
                if col_month_pairs:
                    months = [m for _, m in col_month_pairs]
                    colnames = [c for c, _ in col_month_pairs]
                    values = [row.get(c, np.nan) for c in colnames]
                    pollutant_series[pb] = {"months": months, "cols": colnames, "values": values}

            if not pollutant_series:
                st.warning("No pollutant month-columns found for this site. Showing raw row.")
                # ensure DataFrame for consistent .astype
                if isinstance(row, pd.Series):
                    raw_df = row.to_frame().T
                else:
                    raw_df = pd.DataFrame([row])
                st.dataframe(raw_df.astype(str))
            else:
                # multiselect pollutants to plot
                available = list(pollutant_series.keys())
                chosen = st.multiselect("Chart pollutants", options=available, default=available[:2] if available else [])
                figures = []
                for pb in chosen:
                    obj = pollutant_series[pb]
                    months = obj["months"]
                    vals = [None if pd.isna(x) else float(x) for x in obj["values"]]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=months, y=vals, name=f"{pb}"))
                    fig.add_trace(go.Scatter(x=months, y=vals, mode="lines+markers", name=f"{pb} trend"))
                    fig.update_layout(title=f"{pb} — monthly values", yaxis_title=pb, template="plotly_white", height=320)
                    st.plotly_chart(fig, use_container_width=True)
                    figures.append((pb, fig))

                    # percent change Nov->Jan if available
                    valid_pairs = [(i, v) for i, v in enumerate(vals) if v is not None]
                    if len(valid_pairs) >= 2:
                        a_idx, a_val = valid_pairs[0]
                        b_idx, b_val = valid_pairs[-1]
                        pc = percent_change(a_val, b_val)
                        if not np.isnan(pc):
                            st.write(f"**{pb}**: {pc:.1f}% change from {months[a_idx]} → {months[b_idx]}.")
                        else:
                            st.write(f"**{pb}**: percent change undefined (initial value 0 or missing).")
                    else:
                        st.write(f"**{pb}**: not enough data to compute percent change.")

                # Summary table safe display (string-cast to avoid pyarrow errors)
                summary_rows = []
                for pb, obj in pollutant_series.items():
                    for colname, mm, val in zip(obj["cols"], obj["months"], obj["values"]):
                        summary_rows.append({"metric": f"{pb}_{mm}", "value": val})
                summary_df = pd.DataFrame(summary_rows)
                st.markdown("**Numeric summary**")
                st.dataframe(summary_df.astype(str))

                # Simple interpretation with example thresholds
                example_thresholds = {"Nitrate": 50, "TotalBOD": 3, "TotalCOD": 250, "pH": (6.5, 8.5)}
                interpretation = []
                for pb, obj in pollutant_series.items():
                    vals = [v for v in obj["values"] if pd.notna(v)]
                    if not vals:
                        continue
                    mean_val = np.mean(vals)
                    thr = example_thresholds.get(pb)
                    if thr:
                        if isinstance(thr, tuple):
                            lo, hi = thr
                            if mean_val < lo or mean_val > hi:
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
                    for s in interpretation:
                        st.write("- " + s)

                # Export options
                st.markdown("---")
                site_rows = df[df["site_id"].astype(str) == str(row.get("site_id"))]
                if site_rows.empty:
                    site_rows = pd.DataFrame([row]) if isinstance(row, pd.Series) else pd.DataFrame([row])

                csv_bytes = site_rows.to_csv(index=False).encode("utf-8")
                st.download_button("Download site CSV", data=csv_bytes, file_name=f"{row.get('site_id')}_data.csv", mime="text/csv")

                # Combined PNG export (guarded)
                if figures:
                    combined = go.Figure()
                    for pb, fig in figures:
                        for tr in fig.data:
                            combined.add_trace(tr)
                    combined.update_layout(title=f"{row.get('site_id')} - selected pollutants", template="plotly_white", height=480)
                    try:
                        if allow_png:
                            img_bytes = combined.to_image(format="png", scale=2)
                            st.download_button("Download combined chart PNG", data=img_bytes, file_name=f"{row.get('site_id')}_chart.png", mime="image/png")
                        else:
                            st.info("PNG export disabled (toggle in sidebar).")
                    except Exception as e:
                        st.warning("PNG export failed (kaleido may not be installed).")
                        st.write(f"PNG export error: {e}")

            # raw rows display - ensure DataFrame type and safe string-cast to avoid Arrow errors
            st.markdown("**Raw CSV rows (string-casted for display)**")
            if isinstance(site_rows, pd.Series):
                site_rows = site_rows.to_frame().T
            st.dataframe(site_rows.astype(str))

# Footer
st.markdown("---")
st.markdown("Notes: pollutant detection supports both `Nov_pH` and `pH_Nov` styles. Display tables are cast to strings to avoid PyArrow errors in Streamlit. Customize thresholds and column normalization as needed.")

