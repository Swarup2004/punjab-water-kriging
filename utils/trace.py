import folium

def show_trace(m, gdf):
    gdf_sorted = gdf.sort_values("distance_m")
    coords = list(zip(gdf_sorted.lat, gdf_sorted.lon))
    folium.PolyLine(
        coords,
        color="green",
        weight=3,
        tooltip="Upstream → Downstream"
    ).add_to(m)
