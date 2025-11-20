import matplotlib.cm as cm
import matplotlib.colors as colors

def pollutant_color(value, series):
    norm = colors.Normalize(series.min(), series.max())
    cmap = cm.get_cmap("YlOrRd")
    r, g, b, _ = cmap(norm(value))
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
