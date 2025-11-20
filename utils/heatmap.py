from folium.plugins import HeatMap

def add_heatmap(m, gdf, pollutant, opacity):
    heat_data = [
        [row.lat, row.lon, row[pollutant]]
        for _, row in gdf.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=18,
        blur=15,
        min_opacity=opacity,
        max_zoom=18
    ).add_to(m)
