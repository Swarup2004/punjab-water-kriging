#!/usr/bin/env python3
# Saves: data/nchoe_river_line.geojson, data/nchoe_river_samples.geojson, data/nchoe_river_map.html

import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import numpy as np
import folium

PLACE = "Chandigarh, India"
TAGS = {"waterway": ["river", "stream"]}
NAMES_OF_INTEREST = ["N Choe", "N-Choe", "Attawa Choa", "Attawa Choe", "N-CHOE", "N CHOE"]

SAMPLE_EVERY_METERS = 50  # spacing along river
OUT_DIR = "data"
LINE_OUT = os.path.join(OUT_DIR, "nchoe_river_line.geojson")
PTS_OUT  = os.path.join(OUT_DIR, "nchoe_river_samples.geojson")
MAP_OUT  = os.path.join(OUT_DIR, "nchoe_river_map.html")

def ensure_dir(p): os.makedirs(os.path.dirname(p), exist_ok=True)

def to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Chandigarh ≈ EPSG:32643 (UTM 43N)
    return gdf.to_crs(epsg=32643)

def from_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=4326)

def main():
    ensure_dir(LINE_OUT)
    ensure_dir(PTS_OUT)
    ensure_dir(MAP_OUT)

    # Query OSM
    #gdf = ox.geometries_from_place(PLACE, tags=TAGS)
    if hasattr(ox, "features_from_place"):
    gdf = ox.features_from_place(PLACE, tags=TAGS)          # OSMnx ≥ 2.0
    else:
    gdf = ox.geometries_from_place(PLACE, tags=TAGS)
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    # Filter by name (fallback to all)
    if "name" in gdf.columns:
        rivers = gdf[gdf["name"].astype(str).str.lower().isin([n.lower() for n in NAMES_OF_INTEREST])].copy()
    else:
        rivers = gdf.iloc[0:0].copy()
    if rivers.empty:
        print("No named rivers matched; using all rivers/streams as fallback.")
        rivers = gdf.copy()

    # Merge to single longest line (in meters CRS)
    rivers = rivers.set_crs(epsg=4326, allow_override=True)
    rivers_m = to_utm(rivers)
    merged = linemerge(rivers_m.unary_union)
    if isinstance(merged, LineString):
        line_m = merged
    elif isinstance(merged, MultiLineString):
        line_m = max(list(merged), key=lambda l: l.length)
    else:
        raise ValueError(f"Unexpected geometry type after merge: {type(merged)}")

    # Sample evenly along the line
    total_len = line_m.length
    n_samples = max(int(total_len // SAMPLE_EVERY_METERS) + 1, 2)
    distances = np.linspace(0, total_len, n_samples)
    pts_m = [line_m.interpolate(d) for d in distances]

    # Save GeoJSON (back to WGS84)
    gpd.GeoDataFrame(geometry=[line_m], crs="EPSG:32643").pipe(from_utm).to_file(LINE_OUT, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=pts_m, crs="EPSG:32643").pipe(from_utm).to_file(PTS_OUT, driver="GeoJSON")

    # Quick preview map
    pts_wgs = gpd.GeoDataFrame(geometry=pts_m, crs="EPSG:32643").pipe(from_utm)
    center = [pts_wgs.geometry.y.mean(), pts_wgs.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13)
    folium.GeoJson(gpd.GeoDataFrame(geometry=[line_m], crs="EPSG:32643").pipe(from_utm).__geo_interface__,
                   name="N-CHOE line").add_to(m)
    for pt in pts_wgs.geometry.iloc[::20]:
        folium.CircleMarker(location=[pt.y, pt.x], radius=3, color="blue", fill=True).add_to(m)
    m.save(MAP_OUT)

    print(f"Saved: {LINE_OUT}")
    print(f"Saved: {PTS_OUT}")
    print(f"Saved: {MAP_OUT}")

if __name__ == "__main__":
    main()
