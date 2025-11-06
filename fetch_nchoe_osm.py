#!/usr/bin/env python3
"""
Fetch N-CHOE (Chandigarh, India) waterway linework from OpenStreetMap, merge it to the
longest continuous line, sample points along it, and write:
  - data/nchoe_river_line.geojson
  - data/nchoe_river_samples.geojson
  - data/nchoe_river_map.html

Run:
    python fetch_nchoe_osm.py
"""

import os
import sys
import json
import warnings
from typing import Iterable, List

import numpy as np
import folium
import geopandas as gpd
import osmnx as ox
from shapely.geometry import (
    LineString,
    MultiLineString,
    Point,
    GeometryCollection,
    mapping,
)
from shapely.ops import linemerge


# ---------------------------
# Config
# ---------------------------
PLACE = "Chandigarh, India"
TAGS = {"waterway": ["river", "stream"]}

# Common local names in OSM for the same channel
NAMES_OF_INTEREST = [
    "N Choe",
    "N-Choe",
    "Attawa Choa",
    "Attawa Choe",
    "N-CHOE",
    "N CHOE",
]

# Chandigarh ~ UTM zone 43N
EPSG_WGS84 = 4326
EPSG_UTM43N = 32643

SAMPLE_EVERY_METERS = 50  # distance between sample points

OUT_DIR = "data"
LINE_OUT = os.path.join(OUT_DIR, "nchoe_river_line.geojson")
PTS_OUT = os.path.join(OUT_DIR, "nchoe_river_samples.geojson")
MAP_OUT = os.path.join(OUT_DIR, "nchoe_river_map.html")


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir_for(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project to UTM for metric operations."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=EPSG_WGS84, allow_override=True)
    return gdf.to_crs(epsg=EPSG_UTM43N)


def from_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project back to WGS84 (lat/lon)."""
    return gdf.to_crs(epsg=EPSG_WGS84)


def fetch_osm_features(place: str, tags: dict) -> gpd.GeoDataFrame:
    """
    Works on OSMnx 2.x (features_from_place) and 1.x (geometries_from_place).
    """
    if hasattr(ox, "features_from_place"):
        return ox.features_from_place(place, tags=tags)  # OSMnx ≥2.0
    if hasattr(ox, "geometries_from_place"):
        return ox.geometries_from_place(place, tags=tags)  # OSMnx 1.x
    raise RuntimeError(
        "Your OSMnx version lacks both features_from_place and geometries_from_place."
    )


def pick_river_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to linework; prefer named features matching NAMES_OF_INTEREST, fallback to all."""
    # Keep only linear features
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    if gdf.empty:
        raise ValueError("No linear waterway geometries returned by OSM.")

    # Normalize CRS
    if gdf.crs is None:
        gdf.set_crs(epsg=EPSG_WGS84, inplace=True, allow_override=True)
    else:
        gdf = gdf.to_crs(epsg=EPSG_WGS84)

    # Name-based filtering
    rivers = gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    if "name" in gdf.columns:
        wanted = {n.lower() for n in NAMES_OF_INTEREST}
        name_lower = gdf["name"].astype(str).str.lower()
        rivers = gdf[name_lower.isin(wanted)].copy()

    if rivers.empty:
        print("No named rivers matched; using all rivers/streams as fallback.")
        rivers = gdf.copy()

    return rivers


def _flatten_lines(geom) -> Iterable[LineString]:
    """Yield LineStrings from (possibly nested) Shapely geometry."""
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield g
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from _flatten_lines(g)


def merge_to_longest_line(rivers_wgs: gpd.GeoDataFrame) -> LineString:
    """Union + linemerge, then choose the longest LineString robustly (Shapely 2.x safe)."""
    rivers_m = to_utm(rivers_wgs)  # metric CRS
    merged = linemerge(rivers_m.unary_union)

    if isinstance(merged, LineString):
        return merged

    candidates = list(_flatten_lines(merged))
    if not candidates:
        raise ValueError(f"No linework found after merge; got {merged.geom_type}")

    longest = max(candidates, key=lambda l: l.length)
    print(
        f"Merge produced {len(candidates)} segments; choosing longest: {longest.length:.1f} m."
    )
    return longest


def sample_along_line(line_m: LineString, step_m: float) -> List[Point]:
    total_len = line_m.length
    n_samples = max(int(total_len // step_m) + 1, 2)
    distances = np.linspace(0, total_len, n_samples)
    return [line_m.interpolate(d) for d in distances]


def save_geojson_gdf(gdf: gpd.GeoDataFrame, path: str) -> None:
    """Save a GeoDataFrame as GeoJSON, with a JSON fallback if fiona is unavailable."""
    try:
        gdf.to_file(path, driver="GeoJSON")
        return
    except Exception as e:
        # Fallback: write via __geo_interface__
        print(f"to_file failed ({e}); falling back to raw GeoJSON write.")
        ensure_dir_for(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gdf.__geo_interface__, f)


def make_preview_map(line_m: LineString, pts_m: List[Point], out_html: str) -> None:
    line_wgs = gpd.GeoDataFrame(geometry=[line_m], crs=f"EPSG:{EPSG_UTM43N}")
    line_wgs = from_utm(line_wgs)

    pts_wgs = gpd.GeoDataFrame(geometry=pts_m, crs=f"EPSG:{EPSG_UTM43N}")
    pts_wgs = from_utm(pts_wgs)

    center = [float(pts_wgs.geometry.y.mean()), float(pts_wgs.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=13, control_scale=True)

    folium.GeoJson(
        line_wgs.__geo_interface__, name="N-CHOE river (longest merged line)"
    ).add_to(m)

    # Drop a sparse subset of points to keep map light
    for pt in pts_wgs.geometry.iloc[::20]:
        folium.CircleMarker(
            location=[pt.y, pt.x], radius=3, fill=True
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_html)


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    for p in (LINE_OUT, PTS_OUT, MAP_OUT):
        ensure_dir_for(p)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        gdf = fetch_osm_features(PLACE, TAGS)
        rivers = pick_river_lines(gdf)
        line_m = merge_to_longest_line(rivers)
        pts_m = sample_along_line(line_m, SAMPLE_EVERY_METERS)

        # Save line & points as GeoJSON (WGS84)
        line_wgs = gpd.GeoDataFrame(geometry=[line_m], crs=f"EPSG:{EPSG_UTM43N}")
        line_wgs = from_utm(line_wgs)
        save_geojson_gdf(line_wgs, LINE_OUT)

        pts_wgs = gpd.GeoDataFrame(geometry=pts_m, crs=f"EPSG:{EPSG_UTM43N}")
        pts_wgs = from_utm(pts_wgs)
        save_geojson_gdf(pts_wgs, PTS_OUT)

        # Preview map
        make_preview_map(line_m, pts_m, MAP_OUT)

    print(f"Saved: {LINE_OUT}")
    print(f"Saved: {PTS_OUT}")
    print(f"Saved: {MAP_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
