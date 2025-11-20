import folium

def add_hotspot_layer(m, gdf, pollutant):
    vals = gdf[pollutant]
    thresh = vals.mean() + vals.std()

    hotspots = gdf[gdf[pollutant] > thresh]

    for _, r in hotspots.iterrows():
        folium.Marker(
            [r.lat, r.lon],
            icon=folium.Icon(color="red", icon="warning"),
            tooltip=f"Hotspot: {pollutant} = {r[pollutant]:.2f}"
        ).add_to(m)
