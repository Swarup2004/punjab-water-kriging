import folium
import numpy as np

def add_flow_arrows(m, gdf, angle_col="flow_direction_deg"):
    """
    Draw small arrows in the local flow direction.

    Parameters
    ----------
    m : folium.Map
    gdf : GeoDataFrame with at least 'lat', 'lon' and angle_col (in degrees)
    angle_col : str, name of the column with flow direction in degrees

    If angle_col is missing, this function quietly does nothing.
    """
    if angle_col not in gdf.columns:
        # silently skip if we don't have a flow direction column
        return

    for _, r in gdf.iterrows():
        lat = r["lat"]
        lon = r["lon"]
        angle = r[angle_col]

        # small offset: ~20 m-ish depending on lat
        dx = 0.0002 * np.cos(np.radians(angle))
        dy = 0.0002 * np.sin(np.radians(angle))

        folium.PolyLine(
            [(lat, lon), (lat + dy, lon + dx)],
            color="blue",
            weight=2,
            opacity=0.7
        ).add_to(m)
