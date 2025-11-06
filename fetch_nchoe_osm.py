#!/usr/bin/env python3
"""
Fetch N-CHOE (and related names) river geometry from OSM for Chandigarh,
merge to a single line, sample evenly-spaced points ALONG the river in meters,
and save GeoJSON + an HTML preview map.

Outputs:
  assets/nchoe_river_line.geojson      # merged LineString (EPSG:4326)
  assets/nchoe_river_samples.geojson   # sampled points along the line (EPSG:4326)
  assets/nchoe_river_map.html          # quick Folium preview

Run:
  python scripts/fetch_nchoe_osm.py
"""

import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import numpy as np
import folium

# ---------------- Config ----------------
PLACE = "Chandigarh, India"
TAGS = {"waterway": ["river", "stream"]}
NAMES_OF_INTEREST = ["N Choe", "N-Choe", "Attawa Choa", "Attawa Choe", "N-CHOE", "N CHOE"]

SAMPLE_EVERY_METERS = 50  # ~50 m spacing along the river
ASSETS_DIR = "assets"
LINE_OUT = os.path.join(ASSETS_DIR, "nchoe_river_line.geojson")
PTS_OUT = os.path.join(ASSETS_DIR, "nchoe_river_samples.geojson")
MAP_OUT = os.path.join(ASSETS_DIR, "nchoe_river_map.html")
# ---------------------------------------


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project to a local UTM (metric) CRS for accurate lengths/sampling."""
    # Chandigarh ~ 76.78E, 30.73N → UTM zone 43N (EPSG:32643)
    return gdf.to_crs(epsg=32643)


def from_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=4326)


def main():
    ensure_dir(LINE_OUT)
    ensure_dir(PTS_OUT)
    ensure_dir(MAP_OUT)

    # Step 1: query OSM
    gdf = ox.geometries_from_place(PLACE, tags=TAGS)

    # Keep only line geometries
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    # Step 2: name filter (case-insensitive, keep fallback if empty)
    name_col = "name"
    if name_col in gdf.columns:
        rivers = gdf[gdf[name_col].astype(str).str.lower().isin([n.lower() for n in NAMES_OF_INTEREST])].copy()
    else:
        rivers = gdf.iloc[0:0].copy()

    if rivers.empty:
        print("No named rivers matched; using all rivers/streams in the place as fallback.")
        rivers = gdf.copy()

    # Step 3: merge to a single line (prefer longest) in metric CRS for robust length ops
    rivers = rivers.set_crs(epsg=4326, allow_override=True)
    rivers_m = to_utm(rivers)

    # Merge/union then reduce to a single LineString (longest path)
    merged = linemerge(rivers_m.unary_union)
    if isinstance(merged, LineString):
        line_m = merged
    elif isinstance(merged, MultiLineString):
        line_m = max(list(merged), key=lambda l: l.length)
    else:
        raise ValueError(f"Unexpected geometry type after merge: {type(merged)}")

    # Step 4: sample evenly along the line by distance (meters)
    total_len = line_m.length
    n_samples = max(int(total_len // SAMPLE_EVERY_METERS) + 1, 2)
    distances = np.linspace(0, total_len, n_samples)
    pts_m = [line_m.interpolate(d) for d in distances]

    # Wrap into GeoDataFrames
    line_gdf_m = gpd.GeoDataFrame(geometry=[line_m], crs="EPSG:32643")
    pts_gdf_m = gpd.GeoDataFrame(geometry=pts_m, crs="EPSG:32643")

    # Reproject back to WGS84 (lat/lon)
    line_gdf = from_utm(line_gdf_m)
    pts_gdf = from_utm(pts_gdf_m)

    # Step 5: save GeoJSONs
    line_gdf.to_file(LINE_OUT, driver="GeoJSON")
    pts_gdf.to_file(PTS_OUT, driver="GeoJSON")

    print(f"Saved river line to: {LINE_OUT}")
    print(f"Saved sampled points to: {PTS_OUT} (spacing ~{SAMPLE_EVERY_METERS} m)")

    # Step 6: quick Folium preview
    center = [pts_gdf.geometry.y.mean(), pts_gdf.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13)
    folium.GeoJson(line_gdf.__geo_interface__, name="N-CHOE line").add_to(m)
    # Plot every ~20th point for clarity
    for pt in pts_gdf.geometry.iloc[::20]:
        folium.CircleMarker(location=[pt.y, pt.x], radius=3, color="blue", fill=True).add_to(m)
    m.save(MAP_OUT)
    print(f"Preview map saved to: {MAP_OUT}")


if __name__ == "__main__":
    main()
