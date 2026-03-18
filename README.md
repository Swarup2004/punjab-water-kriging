# Punjab Water Kriging (HydroKriging Pollution Visualizer)

A small geospatial + ML project that:
1. **Interpolates river water-quality measurements** using **Ordinary Kriging** to create predictions along a river.
2. **Visualizes the predicted pollution surface** in an interactive **Streamlit + Folium** web app (points, heatmap, flow arrows, tracing, and simple hotspot detection).
3. Includes an optional utility to **fetch river linework from OpenStreetMap (OSM)** and sample points along it.

---

## What’s in this repo

- **`app.py`** — Streamlit app: interactive map viewer for `data/hydrokriging_predictions.geojson`.
- **`kridging.py`** — Ordinary Kriging pipeline:
  - reads `data/river_data_cleaned.csv`
  - builds intermediate points along the “river axis” using PCA
  - kriges each numeric pollutant column
  - writes `data/kriging_predictions.csv`
- **`fetch_nchoe_osm.py`** — Fetches waterway linework for *Chandigarh, India* from OSM, merges to the longest continuous line, samples points, and writes GeoJSON + an HTML preview map.
- **`utils/`** — Helper modules for map layers (heatmap, arrows, trace line, etc.)
- **`data/`** — GeoJSON/CSV inputs + outputs used by the app and kriging script.
- **`.github/workflows/`** — GitHub Actions workflows to re-run OSM fetch and kriging and commit outputs back to the repo.

---

## Quickstart

### 1) Create an environment + install dependencies

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> Notes:
> - `geopandas`, `fiona`, `shapely`, and friends can require system dependencies depending on OS.
> - If install is painful locally, you can also run the workflows in GitHub Actions.

---

## Run the Streamlit app

```bash
streamlit run app.py
```

The app loads:
- `data/hydrokriging_predictions.geojson` (kriged points & pollutant columns)
- optionally `data/river_polyline.geojson` (river line overlay)

### App features
- Select pollutant column from sidebar
- Toggle:
  - kriged points
  - heatmap overlay
  - flow arrows (requires a `flow_direction_deg` column)
  - upstream→downstream trace (requires a `distance_m` column)
  - “industry hotspots” (simple threshold: mean + std dev)
- Download the currently loaded dataset as CSV

---

## Generate kriging predictions (CLI)

`kridging.py` performs Ordinary Kriging for each numeric pollutant column in your CSV.

### Input
- `data/river_data_cleaned.csv`
  - must include either:
    - `Latitude` and `Longitude` columns, **or**
    - a “Sample Location” column containing text like `"lat, lon"`

### Run

```bash
python kridging.py --infile data/river_data_cleaned.csv --outfile data/kriging_predictions.csv --n 150
```

Options:
- `--n` number of intermediate points sampled along the estimated river axis
- `--variogram` variogram model (`spherical`, `linear`, `gaussian`, `exponential`, etc.)

Output:
- `data/kriging_predictions.csv` with columns like:
  - `Latitude`, `Longitude`
  - `pred_<pollutant_column_name>`

---

## Fetch river geometry from OpenStreetMap (optional)

This script is configured for the **N-CHOE** channel in **Chandigarh, India**.

```bash
python fetch_nchoe_osm.py
```

Outputs (in `data/`):
- `nchoe_river_line.geojson`
- `nchoe_river_samples.geojson`
- `nchoe_river_map.html` (quick preview)

---

## GitHub Actions workflows

Located in `.github/workflows/`:

- **`kriging.yml`**
  - Runs kriging when `kridging.py` or `data/river_data_cleaned.csv` changes (or manually)
  - Commits updated `data/kriging_predictions.csv`

- **`fetch-nchoe-river.yml` / `nchoe-fetch.yml`**
  - Manual workflow to fetch OSM river geometry + samples
  - Commits updated GeoJSON/HTML outputs in `data/`

---

## Data notes

This repo currently includes several large GeoJSON/CSV artifacts under `data/` used by the app. If you plan to expand this project, you may want to:
- document the schema of `hydrokriging_predictions.geojson` (required columns: `lat`, `lon`, plus pollutant columns)
- move large artifacts to Git LFS or external storage if they grow

---

## License

No license file found in the repository yet. If you want, add a `LICENSE` (MIT/Apache-2.0/GPL/etc.) so others know how they can use the code and data.
