#!/usr/bin/env python3
# Saves: data/nchoe_river_line.geojson, data/nchoe_river_samples.geojson, data/nchoe_river_map.html

import os
import warnings

import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
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

def ensure_dir_for(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Chandigarh ≈ EPSG:32643 (UTM 43N)
    return gdf.to_crs(epsg=32643)

def from_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=4326)

def fetch_osm_features(place: str, tags: dict) -> gpd.GeoDataFrame:
    """
    Works with OSMnx 2.x (features_from_place) and 1.x (geometries_from_place).
    """
    if hasattr(ox, "features_from_place"):
        return ox.features_from_place(place, tags=tags)  # OSMnx ≥ 2.0
    elif hasattr(ox, "geometries_from_place"):
        return ox.geometries_from_place(place, tags=tags)  # OSMnx 1.x
    else:
        raise RuntimeError("Your OSMnx version lacks both features_from_place and geometries_from_place")

def pick_river_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Keep only linework
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    # If names exist, try to match by target names
    rivers = gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    if "name" in gdf.columns:
        wanted = {n.lower() for n in NAMES_OF_INTEREST}
        name_lower = gdf["name"].astype(str).str.lower()
        rivers = gdf[name_lower.isin(wanted)].copy()

    if rivers.empty:
        print("No named rivers matched; using all rivers/streams as fallback.")
        rivers = gdf.copy()

    if rivers.empty:
        raise ValueError("No line geometries found from OSM for the given place/tags.")

    # Ensure CRS is WGS84 before further ops
    if rivers.crs is None:
        rivers.set_crs(epsg=4326, inplace=True, allow_override=True)
    else:
        rivers = rivers.to_crs(epsg=4326)

    return rivers

def merge_to_longest_line(rivers_wgs: gpd.GeoDataFrame) -> LineString:
    rivers_m = to_utm(rivers_wgs)
    merged = linemerge(rivers_m.unary_union)

    if isinstance(merged, LineString):
        return merged
    if isinstance(merged, MultiLineString):
        return max(list(merged), key=lambda l: l.length)

    raise ValueError(f"Unexpected geometry type after merge: {type(merged)}")

def sample_along_line(line_m: LineString, step_m: float) -> list[Point]:
    total_len = line_m.length
    n_samples = max(int(total_len // step_m) + 1, 2)
    distances = np.linspace(0, total_len, n_samples)
    return [line_m.interpolate(d) for d in distances]

def make_preview_map(line_m: LineString, pts_m: list[Point], out_html: str) -> None:
    line_wgs = gpd.GeoDataFrame(geometry=[line_m], crs="EPSG:32643").pipe(from_utm)
    pts_wgs  = gpd.GeoDataFrame(geometry=pts_m,  crs="EPSG:32643").pipe(from_utm)

    center = [float(pts_wgs.geometry.y.mean()), float(pts_wgs.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=13, control_scale=True)
    folium.GeoJson(line_wgs.__geo_interface__, name="N-CHOE line").add_to(m)
    for pt in pts_wgs.geometry.iloc[::20]:
        folium.CircleMarker(location=[pt.y, pt.x], radius=3, color=None, fill=True).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_html)

def main():
    for p in (LINE_OUT, PTS_OUT, MAP_OUT):
        ensure_dir_for(p)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        gdf = fetch_osm_features(PLACE, TAGS)
        rivers = pick_river_lines(gdf)
        line_m = merge_to_longest_line(rivers)
        pts_m  = sample_along_line(line_m, SAMPLE_EVERY_METERS)

        # Save GeoJSON (back to WGS84)
        gpd.GeoDataFrame(geometry=[line_m], crs="EPSG:32643").pipe(from_utm).to_file(LINE_OUT, driver="GeoJSON")
        gpd.GeoDataFrame(geometry=pts_m,  crs="EPSG:32643").pipe(from_utm).to_file(PTS_OUT,  driver="GeoJSON")

        # Preview map
        make_preview_map(line_m, pts_m, MAP_OUT)

    print(f"Saved: {LINE_OUT}")
    print(f"Saved: {PTS_OUT}")
    print(f"Saved: {MAP_OUT}")

if __name__ == "__main__":
    main()
