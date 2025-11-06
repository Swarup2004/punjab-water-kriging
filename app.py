# app_kriging_dem.py
"""
Punjab River Pollutant Visualizer — Kriging + DEM + OSM + Upstream Tracing

Run:
    streamlit run app_kriging_dem.py

Notes:
- Set DATA_PATH_1 to your Punjab CSV or keep default (example placeholder).
- Optionally set DATA_PATH_2 to a longer time‑series dataset (Ganga) for modeling.
- DEM can be auto-downloaded (SRTM 30m) for the Punjab bbox if 'elevation' is installed,
  or you can point to a local GeoTIFF (EPSG:4326 or 3857 preferred).
- If PyKrige isn't installed, the app will fall back to IDW interpolation.
- If richdem isn't installed, upstream tracing will be disabled.
"""

import os
from typing import List, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

import plotly.graph_objects as go
import plotly.express as px

# Optional / hydrology + kriging libs (handled gracefully if missing)
try:
    from pykrige.ok import OrdinaryKriging
    _HAS_PYKRIGE = True
except Exception:
    _HAS_PYKRIGE = False

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio import warp as rio_warp
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    import richdem as rd
    _HAS_RICHDEM = True
except Exception:
    _HAS_RICHDEM = False

try:
    import osmnx as ox
    _HAS_OSMNX = True
except Exception:
    _HAS_OSMNX = False

# Optional DEM auto-download (SRTM) helper
try:
    import elevation  # downloads DEM tiles
    _HAS_ELEVATION = True
except Exception:
    _HAS_ELEVATION = False

st.set_page_config(layout="wide", page_title="Punjab River Pollutant Visualizer — Kriging & DEM")

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH_1 = "./data/river_data_cleaned.csv"  # your Punjab dataset
# Long time‑series candidate (Ganga stations). Replace with actual path after download.
DATA_PATH_2 = "./data/ganga_long_term.csv"

GEOJSON_PATH = "assets/punjab_districts.geojson"
MAP_CENTER = [30.9, 75.8]
MAP_ZOOM = 7
PUNJAB_BBOX = [28.5, 73.5, 32.5, 77.5]  # S,W,N,E (lat/lon)

# ----------------------------
# UTILS
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize some month tags
    col_map = {}
    for c in df.columns:
        newc = c
        newc = newc.replace("Nov-24_", "Nov_").replace("Dec-24_", "Dec_").replace("Jan-25_", "Jan_")
        newc = newc.replace("Nov-24 ", "Nov_").replace("Dec-24 ", "Dec_").replace("Jan-25 ", "Jan_")
        newc = newc.replace("Nov-24", "Nov").replace("Dec-24", "Dec").replace("Jan-25", "Jan")
        col_map[c] = newc
    df = df.rename(columns=col_map)

    # lat/lon/site
    lat_candidates = [c for c in df.columns if c.lower() in ("lat", "latitude", "latitude (deg)")]
    lon_candidates = [c for c in df.columns if c.lower() in ("lon", "longitude", "long", "longitude (deg)")]
    if lat_candidates:
        df = df.rename(columns={lat_candidates[0]: "lat"})
    if lon_candidates:
        df = df.rename(columns={lon_candidates[0]: "lon"})
    site_candidates = [c for c in df.columns if c.lower() in ("site_id","site","sample location","sample code","sample_location","location")]
    if site_candidates:
        df = df.rename(columns={site_candidates[0]: "site_id"})
    else:
        other = [c for c in df.columns if "site" in c.lower() or "sample" in c.lower() or "location" in c.lower()]
        if other: df = df.rename(columns={other[0]:"site_id"})
        else: df["site_id"] = df.index.astype(str)

    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)
    return df

def detect_pollutants(df: pd.DataFrame) -> List[str]:
    months = ["Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"]
    pollutant_bases = set()
    for col in df.columns:
        for m in months:
            if col.startswith(f"{m}_"):
                pollutant_bases.add(col.split(f"{m}_",1)[1])
            if col.endswith(f"_{m}"):
                pollutant_bases.add(col.rsplit(f"_{m}",1)[0])
    if not pollutant_bases:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pollutant_bases = set([c for c in numeric_cols if c not in ("lat","lon")][:12])
    return sorted(pollutant_bases)

def find_month_cols(df: pd.DataFrame, pb: str, months_list: List[str]) -> List[Tuple[str,str]]:
    out = []
    for m in months_list:
        pref = f"{m}_{pb}"; suff = f"{pb}_{m}"
        if pref in df.columns: out.append((pref,m))
        elif suff in df.columns: out.append((suff,m))
        else:
            for c in df.columns:
                if c.lower()==pref.lower(): out.append((c,m)); break
                if c.lower()==suff.lower(): out.append((c,m)); break
    return out

def compute_agg(df: pd.DataFrame, pb: str, months_sel: List[str]) -> pd.Series:
    pairs = find_month_cols(df, pb, months_sel)
    cols = [c for c,_ in pairs]
    if not cols:
        return pd.Series([0.0]*len(df), index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.max(axis=1, skipna=True).fillna(0)

def idw_interpolate(x, y, z, grid_x, grid_y, power=2, eps=1e-12):
    # Simple IDW for fallback
    z = np.asarray(z, dtype=float)
    xi = grid_x.flatten(); yi = grid_y.flatten()
    out = np.zeros_like(xi, dtype=float)
    for i,(gx,gy) in enumerate(zip(xi, yi)):
        d2 = (x - gx)**2 + (y - gy)**2 + eps
        w = 1.0 / (d2**(power/2))
        out[i] = np.sum(w*z) / np.sum(w)
    return out.reshape(grid_x.shape)

def run_kriging(x, y, z, grid_x, grid_y, variogram_model="spherical"):
    if not _HAS_PYKRIGE:
        return idw_interpolate(x,y,z,grid_x,grid_y)
    OK = OrdinaryKriging(x, y, z, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    zi, ss = OK.execute("grid", grid_x[0,:], grid_y[:,0])
    return np.array(zi)

# ------------- DEM helpers -------------
@st.cache_data(show_spinner=False)
def download_dem(bounds: Tuple[float,float,float,float], out_path: str) -> Optional[str]:
    """bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84"""
    if not _HAS_ELEVATION:
        return None
    try:
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # elevation expects (min_lon, min_lat, max_lon, max_lat)
        elevation.clip(bounds=bounds, output=out_path, product='SRTM1')
        return out_path if os.path.exists(out_path) else None
    except Exception:
        return None

def load_dem(path: str):
    if not (_HAS_RASTERIO and os.path.exists(path)):
        return None, None
    src = rasterio.open(path)
    data = src.read(1, masked=True)
    return src, data

def dem_to_richdem_array(src, data):
    if not _HAS_RICHDEM: return None
    # Ensure data is in meters and no nodata
    arr = np.where(np.ma.getmaskarray(data), np.nan, data.astype("float64"))
    # Replace nans with local mean to avoid sink issues
    mean_val = np.nanmean(arr)
    arr = np.where(np.isnan(arr), mean_val, arr)
    rd_dem = rd.rdarray(arr, no_data=mean_val)
    rd_dem.geotransform = src.transform.to_gdal()
    return rd_dem

def compute_flow(rd_dem):
    if not _HAS_RICHDEM: return None, None
    fdir = rd.FlowDirectionD8(rd_dem)
    facc = rd.FlowAccumulation(fdir, method='D8')
    return fdir, facc

def upstream_mask_to_point(fdir, facc, src, lat, lon, max_cells=2_000_000):
    """Return boolean mask (numpy) of all cells that flow to the clicked cell using reverse D8 tracing."""
    if not (_HAS_RICHDEM and _HAS_RASTERIO): return None
    # Convert lat/lon to pixel
    row, col = ~src.transform * (lon, lat)  # note order (x,y)=(lon,lat)
    row = int(round(row)); col = int(round(col))
    if row<0 or col<0 or row>=facc.shape[0] or col>=facc.shape[1]:
        return None
    # Reverse accumulate: select all cells whose flow path eventually reaches (row,col)
    # Approximate via "contributing area threshold": select facc >= facc[row,col] / k around basin
    # For sharp upstream domain, do a BFS using D8 pointers (costly). We'll do a hybrid threshold + BFS limit.
    target_acc = facc[row, col]
    # Start BFS from target cell backwards: neighbors that point to this cell
    mask = np.zeros_like(facc, dtype=bool)
    stack = [(row, col)]
    count = 0
    # D8 neighbor offsets and reverse codes derived from richdem D8 scheme
    nbrs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    while stack and count < max_cells:
        r,c = stack.pop()
        if mask[r,c]: continue
        mask[r,c] = True; count += 1
        # examine neighbors: if their flow goes to (r,c), include
        for dr,dc in nbrs:
            rr,cc = r+dr, c+dc
            if rr<0 or cc<0 or rr>=facc.shape[0] or cc>=facc.shape[1]: continue
            # If neighbor flows into (r,c): i.e., the steepest descent from (rr,cc) leads to (r,c)
            # We check by following one D8 step from (rr,cc)
            mdir = int(rd.get_d8_flow_directions_to((rr,cc), (fdir[rr,cc]))[0][0]) if True else 0
            # Fallback: compare facc decrease towards (r,c)
            # But richdem doesn't expose a direct pointer; use heuristic: move to neighbor with min elevation; skipped here.
            # Try using richdem's helper to get downstream cell
            try:
                nxt = rd.util.downstream_cell((rr,cc), fdir[rr,cc])
                if nxt == (r,c):
                    stack.append((rr,cc))
            except Exception:
                # heuristic fallback: include if facc[rr,cc] > facc[r,c]
                if facc[rr,cc] > facc[r,c]:
                    stack.append((rr,cc))
    return mask

# ------------- OSM helpers -------------
@st.cache_data(show_spinner=False)
def fetch_osm_gdf(bbox: Tuple[float,float,float,float]):
    """bbox in latlon: (south, west, north, east)"""
    if not _HAS_OSMNX: return None, None, None
    south, west, north, east = bbox
    tags_industry = {"landuse": "industrial"} | {"man_made":["works","industrial"]}
    tags_factories = {"industrial": True, "man_made":["works","factory"]}
    tags_residential = {"landuse": "residential"}

    try:
        gdf_ind = ox.geometries_from_bbox(north, south, east, west, tags_industry)
    except Exception:
        gdf_ind = None
    try:
        gdf_fac = ox.geometries_from_bbox(north, south, east, west, tags_factories)
    except Exception:
        gdf_fac = None
    try:
        gdf_res = ox.geometries_from_bbox(north, south, east, west, tags_residential)
    except Exception:
        gdf_res = None
    return gdf_ind, gdf_fac, gdf_res

# ----------------------------
# LOAD DATA
# ----------------------------
df1 = load_csv(DATA_PATH_1)
pollutant_bases1 = detect_pollutants(df1) if not df1.empty else []

df2 = load_csv(DATA_PATH_2)  # optional long-term dataset (Ganga)
pollutant_bases2 = detect_pollutants(df2) if not df2.empty else []

gdf_districts = None
if os.path.exists(GEOJSON_PATH):
    try:
        gdf_districts = gpd.read_file(GEOJSON_PATH)
    except Exception:
        gdf_districts = None

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("Layers & Settings")

dataset_choice = st.sidebar.selectbox("Dataset", ["Punjab (DATA_PATH_1)"] + (["Ganga long-term (DATA_PATH_2)"] if not df2.empty else []))
active_df = df1 if dataset_choice.startswith("Punjab") else df2
active_pollutants = pollutant_bases1 if dataset_choice.startswith("Punjab") else pollutant_bases2

months_all = ["Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"]
months_selected = st.sidebar.multiselect("Months for aggregates", months_all, default=["Nov","Dec","Jan"])

marker_size_option = st.sidebar.selectbox("Marker size by", ["fixed", "value"], index=1)
fixed_marker_px = st.sidebar.slider("Fixed marker radius (px)", 4, 20, 8)
cluster_toggle = st.sidebar.checkbox("Use marker clustering", value=True)
search_text = st.sidebar.text_input("Search site", value="")

show_districts = st.sidebar.checkbox("Show district boundaries", value=True)
show_osm_places = st.sidebar.checkbox("Show OSM places (industrial / factories / residential)", value=False)

# DEM controls
st.sidebar.subheader("DEM / Hydrology")
use_dem = st.sidebar.checkbox("Enable DEM layers & upstream tracing", value=False)
dem_source = st.sidebar.selectbox("DEM source", ["Auto-download SRTM (needs 'elevation')", "Local GeoTIFF path"])
dem_local_path = st.sidebar.text_input("DEM GeoTIFF path (if Local)", value="assets/dem_punjab.tif")
shade_relief = st.sidebar.checkbox("Show hillshade (approx)", value=True)

# Kriging controls
st.sidebar.subheader("Interpolation")
krig_pollutant = st.sidebar.selectbox("Interpolate pollutant", active_pollutants if active_pollutants else ["(none)"])
variogram_model = st.sidebar.selectbox("Variogram", ["spherical","exponential","gaussian"])
grid_res_km = st.sidebar.slider("Grid resolution (km)", min_value=1, max_value=20, value=5)
div_colormap = st.sidebar.selectbox("Diverging colormap (matplotlib)", ["RdBu_r","BrBG","PiYG","PuOr","coolwarm","seismic"])
show_diff_layer = st.sidebar.checkbox("Show differential (downstream deltas)", value=True)

# ----------------------------
# MAIN
# ----------------------------
st.title("Punjab River Pollutant Visualizer — Kriging • DEM • OSM • Upstream")

if active_df.empty:
    st.error("Active dataset is empty. Please update DATA_PATHs at top of script.")
    st.stop()

# Build Map
map_col, right_col = st.columns([2.3, 1])

with map_col:
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, control_scale=True)
    m.fit_bounds([[PUNJAB_BBOX[0], PUNJAB_BBOX[1]], [PUNJAB_BBOX[2], PUNJAB_BBOX[3]]])

    if gdf_districts is not None and show_districts:
        folium.GeoJson(gdf_districts.to_json(), name="districts",
                       style_function=lambda feat: {"fillColor":"#ffffff00","color":"#333","weight":1,"opacity":0.6}).add_to(m)

    # OSM layers
    gdf_ind=gdf_fac=gdf_res=None
    if show_osm_places and _HAS_OSMNX:
        with st.spinner("Fetching OSM places…"):
            gdf_ind,gdf_fac,gdf_res = fetch_osm_gdf(tuple(PUNJAB_BBOX))
        def add_gdf_points(gdf, name, color):
            if gdf is None or gdf.empty: return
            if not isinstance(gdf, gpd.GeoDataFrame): return
            gdf_points = gdf[gdf.geometry.type.isin(["Point","MultiPoint"])]
            for _,r in gdf_points.iterrows():
                geom = r.geometry
                if geom.geom_type=="MultiPoint":
                    pts = list(geom.geoms)
                else:
                    pts = [geom]
                for pt in pts:
                    folium.CircleMarker(location=[pt.y, pt.x], radius=3, color=color, fill=True, fill_color=color, fill_opacity=0.9,
                                        tooltip=folium.Tooltip(f"{name}")).add_to(m)
        add_gdf_points(gdf_ind, "Industrial area", "purple")
        add_gdf_points(gdf_fac, "Factory", "black")
        add_gdf_points(gdf_res, "Residential", "orange")

    # Active DF filter/search
    if search_text.strip():
        filtered_df = active_df[active_df["site_id"].astype(str).str.contains(search_text, case=False, na=False)]
    else:
        filtered_df = active_df

    # Aggregate for marker color
    if krig_pollutant and krig_pollutant != "(none)":
        agg_series = compute_agg(active_df, krig_pollutant, months_selected)
    else:
        agg_series = pd.Series([0.0]*len(active_df), index=active_df.index)

    vmin = float(np.nanmin(agg_series)) if len(agg_series)>0 else 0.0
    vmax = float(np.nanmax(agg_series)) if len(agg_series)>0 else vmin+1.0
    if vmin==vmax: vmax = vmin+1.0

    def v2c(v):
        try:
            frac = (v - vmin)/(vmax-vmin)
            frac = min(max(frac,0.0),1.0)
            return px.colors.sample_colorscale("Viridis",[frac])[0]
        except Exception:
            return "gray"

    marker_cluster = MarkerCluster() if cluster_toggle else None
    if marker_cluster: m.add_child(marker_cluster)

    for idx,row in filtered_df.iterrows():
        lat,lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon): continue
        val = float(agg_series.loc[idx]) if idx in agg_series.index else 0.0
        color = v2c(val)
        radius = float(np.interp(val,[vmin,vmax],[4,18])) if marker_size_option=="value" else fixed_marker_px
        tooltip_lines = [f"<b>{row.get('site_id','')}</b>", f"({lat:.5f}, {lon:.5f})"]
        # show selected pollutant months
        pairs = find_month_cols(active_df, krig_pollutant, months_selected)
        if pairs:
            parts = []
            for c,mm in pairs:
                v = row.get(c)
                if pd.notna(v): parts.append(f"{mm}:{v}")
            if parts:
                tooltip_lines.append(f"{krig_pollutant}: {' / '.join(parts)}")
        marker = folium.CircleMarker([lat,lon], radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.85,
                                     tooltip=folium.Tooltip("<br>".join(tooltip_lines), sticky=True))
        popup_html = f"<div><b>{row.get('site_id','')}</b><br/>Index:{idx}<br/>Coords:{lat:.5f},{lon:.5f}</div>"
        folium.Popup(popup_html, max_width=280).add_to(marker)
        (marker_cluster.add_child(marker) if marker_cluster else marker.add_to(m))

    # ----- DEM loading / hillshade / upstream tracing -----
    dem_src = dem_arr = fdir = facc = None
    if use_dem:
        dem_path = None
        if dem_source.startswith("Auto") and _HAS_ELEVATION:
            dem_path = download_dem((PUNJAB_BBOX[1], PUNJAB_BBOX[0], PUNJAB_BBOX[3], PUNJAB_BBOX[2]), "assets/srtm_punjab.tif")
        elif dem_source.startswith("Local"):
            dem_path = dem_local_path if os.path.exists(dem_local_path) else None

        if dem_path is None:
            st.warning("DEM not available. Install 'elevation' for auto-download or set a valid local GeoTIFF path.")
        else:
            dem_src, dem_arr = load_dem(dem_path)
            if dem_src is None:
                st.warning("Failed to load DEM GeoTIFF.")
            else:
                # crude hillshade
                if shade_relief:
                    try:
                        import matplotlib.colors as mcolors
                        from matplotlib.colors import LightSource
                        ls = LightSource(azdeg=315, altdeg=45)
                        shaded = ls.hillshade(np.array(dem_arr), vert_exag=1, dx=1, dy=1)
                        # overlay via ImageOverlay
                        import PIL.Image as Image
                        img = (shaded*255).astype(np.uint8)
                        from io import BytesIO
                        from base64 import b64encode
                        pil = Image.fromarray(img)
                        buf = BytesIO(); pil.save(buf, format="PNG"); b64 = b64encode(buf.getvalue()).decode()
                        folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{b64}",
                                                          bounds=[[PUNJAB_BBOX[0],PUNJAB_BBOX[1]],[PUNJAB_BBOX[2],PUNJAB_BBOX[3]]],
                                                          opacity=0.45, name="DEM hillshade").add_to(m)
                    except Exception:
                        st.info("Hillshade rendering skipped (Pillow/matplotlib not available).")

                # Hydrology
                rd_dem = dem_to_richdem_array(dem_src, dem_arr)
                if rd_dem is not None and _HAS_RICHDEM:
                    fdir, facc = compute_flow(rd_dem)

    # ----- Kriging surface -----
    if "lat" in active_df.columns and "lon" in active_df.columns and krig_pollutant and krig_pollutant!="(none)":
        st.caption("Interpolating surface (kriging or IDW fallback)…")
        # Build training vectors from selected months' aggregate
        vals = compute_agg(active_df, krig_pollutant, months_selected)
        pts = active_df[["lon","lat"]].to_numpy()  # (x=lon,y=lat)
        # Grid
        # Approx km to degrees (~111km per degree)
        step = grid_res_km/111.0
        gx = np.arange(PUNJAB_BBOX[1], PUNJAB_BBOX[3]+step, step)
        gy = np.arange(PUNJAB_BBOX[0], PUNJAB_BBOX[2]+step, step)
        grid_x, grid_y = np.meshgrid(gx, gy)
        # Filter valid points
        mask = np.isfinite(vals.values)
        x = pts[mask,0]; y = pts[mask,1]; z = vals.values[mask].astype(float)
        try:
            Z = run_kriging(x,y,z,grid_x,grid_y,variogram_model=variogram_model)
            # Normalize to diverging cmap range around median
            med = np.nanmedian(Z)
            # Create colorized PNG overlay
            from matplotlib import cm, colors
            cmap = cm.get_cmap(div_colormap)
            norm = colors.TwoSlopeNorm(vmin=np.nanmin(Z), vcenter=med, vmax=np.nanmax(Z))
            rgba = cmap(norm(Z))
            import PIL.Image as Image
            img = (rgba[:,:,:3]*255).astype(np.uint8)
            from io import BytesIO
            from base64 import b64encode
            pil = Image.fromarray(img)
            buf = BytesIO(); pil.save(buf, format="PNG"); b64 = b64encode(buf.getvalue()).decode()
            folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{b64}",
                                              bounds=[[gy.min(), gx.min()],[gy.max(), gx.max()]],
                                              opacity=0.5, name=f"Kriging: {krig_pollutant}").add_to(m)
        except Exception as e:
            st.warning(f"Interpolation failed: {e}")

    # Legend for point layer
    legend_html = f"""
    <div style="position: fixed; bottom: 45px; left: 10px; z-index:9999;
                background-color: rgba(255,255,255,0.9); padding:8px; border-radius:6px;">
      <b>Legend</b><br>
      Pollutant (points): <b>{krig_pollutant}</b><br>
      Range: {vmin:.2f} — {vmax:.2f}<br>
      Colormap: Viridis (points), {div_colormap} (surface)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Click handling — upstream basin display if DEM available
    map_data = st_folium(m, width="100%", height=680, returned_objects=["last_clicked"])
    last_click = map_data.get("last_clicked") if isinstance(map_data, dict) else None

with right_col:
    # Site selector & charts, plus differential downstream deltas
    st.subheader("Site details / charts / differentials")
    site_list = active_df["site_id"].astype(str).tolist()
    site_dropdown = st.selectbox("Choose a site", options=["(none)"] + site_list, index=0)
    idx_input = st.number_input("Or enter site index (or -1)", min_value=-1, max_value=len(active_df)-1, value=-1, step=1)

    selected_idx = None
    if idx_input >= 0:
        selected_idx = int(idx_input)
    elif site_dropdown != "(none)":
        matches = active_df.index[active_df["site_id"].astype(str) == site_dropdown].tolist()
        selected_idx = matches[0] if matches else None
    elif last_click and last_click.get("lat") is not None and "lat" in active_df.columns:
        lat_click, lon_click = last_click.get("lat"), last_click.get("lng")
        dists = ((active_df["lat"] - lat_click)**2 + (active_df["lon"] - lon_click)**2)
        closest = int(dists.idxmin())
        if dists.min() <= 0.0009:
            selected_idx = closest

    if selected_idx is None:
        st.info("Select a site with the dropdown, index, or click a marker on the map.")
    else:
        if selected_idx not in active_df.index:
            st.error("Index not found in data.")
        else:
            row = active_df.loc[selected_idx]
            st.markdown(f"**{row.get('site_id','Site')}**  \n({row['lat']:.6f}, {row['lon']:.6f})")

            # Build per‑pollutant monthly chart
            pollutant_series = {}
            for pb in active_pollutants:
                pairs = find_month_cols(active_df, pb, months_all)
                if pairs:
                    months = [m for _,m in pairs]
                    cols = [c for c,_ in pairs]
                    values = [row.get(c, np.nan) for c in cols]
                    pollutant_series[pb] = {"months":months, "cols":cols, "values":values}

            to_plot = st.multiselect("Plot pollutants", options=list(pollutant_series.keys()), default=list(pollutant_series.keys())[:2])
            for pb in to_plot:
                obj = pollutant_series[pb]
                months = obj["months"]
                vals = [None if pd.isna(x) else float(x) for x in obj["values"]]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=months, y=vals, name=pb))
                fig.add_trace(go.Scatter(x=months, y=vals, mode="lines+markers", name=f"{pb} trend"))
                fig.update_layout(title=f"{pb} — monthly values", yaxis_title=pb, template="plotly_white", height=300)
                st.plotly_chart(fig, use_container_width=True)

            # ---------- Differential downstream (deltas) ----------
            if show_diff_layer and "lat" in active_df.columns and "lon" in active_df.columns:
                st.markdown("**Differential (downstream deltas)** — each site's value minus the nearest upstream site's value along the DEM flow.")
                # Build simple downstream graph using DEM flow (approximate). If DEM not available, fallback to nearest neighbor by latitude decrease
                if use_dem and _HAS_RICHDEM and fdir is not None and facc is not None and dem_src is not None:
                    # For each site, trace one step downstream (D8) repeatedly until we hit another site; compute delta
                    def latlon_to_rc(lat, lon):
                        r,c = ~dem_src.transform * (lon,lat)
                        return int(round(r)), int(round(c))

                    coords = active_df[["lat","lon"]].to_numpy()
                    rc = np.array([latlon_to_rc(la,lo) for la,lo in coords])

                    # Build map from cell -> station index
                    cell_to_idx = {}
                    for i,(r,c) in enumerate(rc):
                        if 0 <= r < facc.shape[0] and 0 <= c < facc.shape[1]:
                            cell_to_idx[(r,c)] = i

                    def downstream_cell(r,c):
                        # Get next cell from richdem
                        try:
                            rr,cc = rd.util.downstream_cell((r,c), fdir[r,c])
                            return rr,cc
                        except Exception:
                            return r,c

                    # Compute deltas for the chosen pollutant (aggregate over selected months)
                    vals = compute_agg(active_df, krig_pollutant, months_selected).values.astype(float)
                    deltas = np.full(len(active_df), np.nan)
                    for i,(r,c) in enumerate(rc):
                        if not (0 <= r < facc.shape[0] and 0 <= c < facc.shape[1]):
                            continue
                        # walk downstream until we hit another station or edge
                        rr,cc = r,c
                        visited = 0
                        found_down = None
                        while visited < 5000:
                            nrr,ncc = downstream_cell(rr,cc)
                            if (nrr,ncc) == (rr,cc): break
                            rr,cc = nrr,ncc; visited += 1
                            j = cell_to_idx.get((rr,cc), None)
                            if j is not None and j != i:
                                found_down = j
                                break
                        if found_down is None:
                            # leaf outlet: delta = current - 0 (assume no previous)
                            deltas[i] = vals[i]
                        else:
                            deltas[i] = vals[found_down] - vals[i]  # "new load added between i and downstream j"
                    # Display small table for this site and neighbors
                    st.dataframe(pd.DataFrame({"site":active_df["site_id"], "delta_to_downstream":deltas}).astype(str), use_container_width=True)
                else:
                    st.info("Differential deltas need DEM flow. Enable DEM and install 'richdem' for best results.")

            # ---------- Upstream trace for a clicked location ----------
            if last_click and use_dem and _HAS_RICHDEM and dem_src is not None and fdir is not None and facc is not None:
                st.markdown("**Upstream contributors (click on map)**")
                lat_click, lon_click = last_click.get("lat"), last_click.get("lng")
                with st.spinner("Computing upstream contributing area…"):
                    mask = upstream_mask_to_point(fdir, facc, dem_src, lat_click, lon_click)
                if mask is None:
                    st.warning("Upstream mask not available for this click or DEM not aligned.")
                else:
                    # Count stations inside mask as potential contributors
                    def latlon_to_rc(lat, lon):
                        r,c = ~dem_src.transform * (lon,lat); return int(round(r)), int(round(c))
                    rc_sites = np.array([latlon_to_rc(la,lo) for la,lo in active_df[["lat","lon"]].to_numpy()])
                    inside = []
                    for i,(r,c) in enumerate(rc_sites):
                        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r,c]:
                            inside.append(i)
                    contrib_df = active_df.iloc[inside][["site_id","lat","lon"]].copy()
                    st.dataframe(contrib_df.astype(str), use_container_width=True)
                    st.success(f"{len(inside)} monitoring sites lie upstream of the clicked location (approximate).")

# Footer: tips
st.markdown("---")
st.markdown("""
**Tips**
- For DEM features (hillshade, flow, upstream tracing), install: `richdem rasterio elevation` and supply/auto‑download a DEM.
- For Kriging, install: `pykrige` (or the app uses IDW fallback).
- For OSM places layer, install: `osmnx`.
- Diverging colormaps used for surfaces (e.g., RdBu_r, BrBG, PiYG, PuOr, coolwarm, seismic). Points use Viridis.
""")
