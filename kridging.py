#!/usr/bin/env python3
"""
Reads data/river_data_cleaned.csv, does Ordinary Kriging from the available
sample points (assumed to be on one river), and writes predictions for many
intermediate points to data/kriging_predictions.csv.

Usage (from repo root):
    python scripts/kriging_to_data.py --n 150
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from pykrige.ok import OrdinaryKriging
except Exception as e:
    raise SystemExit("Missing dependency: pip install pykrige\n" + str(e))


DATA_IN = Path("data/river_data_cleaned.csv")
DATA_OUT = Path("data/kriging_predictions.csv")


def _try_parse_latlon_str(s):
    if not isinstance(s, str):
        return (np.nan, np.nan)
    m = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if not m:
        return (np.nan, np.nan)
    return float(m.group(1)), float(m.group(2))


def _ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    # Use Latitude/Longitude if present; otherwise derive from "Sample Location"
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        sl_candidates = [c for c in df.columns if "sample location" in c.lower()]
        if sl_candidates:
            lat, lon = zip(*[_try_parse_latlon_str(x) for x in df[sl_candidates[0]].astype(str)])
            df["Latitude"] = pd.to_numeric(lat, errors="coerce")
            df["Longitude"] = pd.to_numeric(lon, errors="coerce")
        else:
            raise ValueError("Could not find Latitude/Longitude or a 'Sample Location' with 'lat, lon' text.")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna(subset=["Latitude", "Longitude"]).copy()


def _pick_pollutants(df: pd.DataFrame):
    exclude = {"latitude", "longitude", "sample code", "sample location"}
    cols = []
    for c in df.columns:
        if c.strip().lower() in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise SystemExit("No numeric pollutant columns found.")
    return cols


def _pca_line_points(lat, lon, n_points=120, expand=0.1):
    # Estimate river direction with PCA on (lon,lat), then sample along that axis
    X = np.column_stack([lon, lat])
    p = PCA(n_components=1).fit(X)
    t = p.transform(X).ravel()
    tmin, tmax = t.min(), t.max()
    span = max(tmax - tmin, 1e-9)
    tmin -= expand * span
    tmax += expand * span
    tq = np.linspace(tmin, tmax, n_points).reshape(-1, 1)
    Xq = p.mean_ + tq @ p.components_[0].reshape(1, 2)
    lon_q = Xq[:, 0]
    lat_q = Xq[:, 1]
    return lat_q, lon_q


def _krige_points(lon, lat, values, lon_q, lat_q, variogram="spherical"):
    m = ~np.isnan(values)
    if m.sum() < 3:
        raise ValueError("Need at least 3 valid samples for kriging.")
    OK = OrdinaryKriging(
        x=lon[m], y=lat[m], z=values[m].astype(float),
        variogram_model=variogram, verbose=False, enable_plotting=False
    )
    zq, _ = OK.execute("points", lon_q, lat_q)
    return np.asarray(zq).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="Number of intermediate points along river")
    ap.add_argument("--variogram", default="spherical",
                    choices=["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"],
                    help="Variogram model")
    ap.add_argument("--infile", default=str(DATA_IN), help="Input CSV path")
    ap.add_argument("--outfile", default=str(DATA_OUT), help="Output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    # Clean any stray text (e.g., '... this is the data ...') in numeric cells
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = (
                df[c].astype(str)
                    .str.replace(r"this is the data.*$", "", regex=True)
                    .str.strip()
            )
            maybe = pd.to_numeric(df[c], errors="ignore")
            df[c] = maybe

    df = _ensure_lat_lon(df)
    if len(df) < 3:
        raise SystemExit(f"Need at least 3 points; found {len(df)}.")

    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)

    # Build intermediate points along the river axis
    lat_q, lon_q = _pca_line_points(lat, lon, n_points=args.n)

    metrics = _pick_pollutants(df)

    out = pd.DataFrame({"Latitude": lat_q, "Longitude": lon_q})
    for metric in metrics:
        vals = pd.to_numeric(df[metric], errors="coerce").to_numpy()
        try:
            out[f"pred_{metric}"] = _krige_points(lon, lat, vals, lon_q, lat_q, variogram=args.variogram)
        except Exception as e:
            print(f"[warn] skipping {metric}: {e}")

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outfile, index=False)
    print(f"[done] wrote {args.outfile}")
    print(f"[info] columns -> {list(out.columns)}")


if __name__ == "__main__":
    main()
