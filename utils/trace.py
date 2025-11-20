import folium

def show_trace(m, gdf, dist_col="distance_m"):
    """
    Draw a line from upstream → downstream.
    Requires a numeric distance column.
    Gracefully skips if the column is missing.
    """
    # If distance column doesn't exist, skip
    if dist_col not in gdf.columns:
        return  # silently skip (app.py will already warn)
    
    try:
        gdf_sorted = gdf.sort_values(dist_col)
    except Exception:
        return

    coords = list(zip(gdf_sorted.lat, gdf_sorted.lon))

    folium.PolyLine(
        coords,
        color="green",
        weight=3,
        tooltip="Upstream → Downstream"
    ).add_to(m)
