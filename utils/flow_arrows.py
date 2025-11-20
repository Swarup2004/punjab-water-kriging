import folium
import numpy as np

def add_flow_arrows(m, gdf):
    for _, r in gdf.iterrows():
        angle = r["flow_direction_deg"]
        lat, lon = r["lat"], r["lon"]

        dx = 0.0002 * np.cos(np.radians(angle))
        dy = 0.0002 * np.sin(np.radians(angle))

        folium.PolyLine(
            [(lat, lon), (lat + dy, lon + dx)],
            color="blue",
            weight=2,
            opacity=0.7
        ).add_to(m)
