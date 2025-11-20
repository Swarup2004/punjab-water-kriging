import folium

def add_river_polyline(m, river_line):
    folium.GeoJson(
        river_line,
        name="River",
        style_function=lambda x: {"color": "blue", "weight": 4}
    ).add_to(m)
