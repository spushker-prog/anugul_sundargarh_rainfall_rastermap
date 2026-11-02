"""
Annual rainfall raster interpolation (CHIRPS pixel centroids -> continuous GeoTIFFs)

Input : CHIRPS pixels CSV  (pixel_id, lon, lat, P_2017 ... P_2025)
        AOI shapefile      (used to clip the output; any CRS)
Output: one GeoTIFF per year + multi-year mean + PNG preview per year

Usage : python rainfall_interpolation.py
        (edit the CONFIG block below first)

Install: pip install pandas numpy scipy geopandas rasterio matplotlib
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
PIXELS_CSV = "Anugul_Sundargarh_CHIRPS_pixels_annual.csv"
AOI_SHP    = "Anugul_sundargarh.shp"          # unzip the zip first
OUT_DIR    = Path("rainfall_rasters")

METHOD     = "idw"      # "idw" | "rbf" | "linear" | "cubic"
CELL_SIZE  = 0.01       # output resolution in degrees (~1.1 km). CHIRPS native = 0.05
IDW_POWER  = 2.0        # IDW only
IDW_NEIGHBORS = 12      # IDW only: nearest points used per cell
RBF_KERNEL = "thin_plate_spline"   # RBF only (also: "multiquadric", "cubic" ...)
CLIP_TO_AOI = True
BUFFER_DEG  = 0.05      # extra margin around AOI so edge cells have data on all sides
NODATA      = -9999.0
OUT_CRS     = "EPSG:4326"
# ----------------------------------------------------------------------------


def idw(xy, z, grid_xy, power=2.0, k=12):
    """Inverse Distance Weighting using k nearest neighbours."""
    tree = cKDTree(xy)
    k = min(k, len(xy))
    dist, idx = tree.query(grid_xy, k=k)
    dist = np.maximum(dist, 1e-12)               # avoid divide-by-zero on exact hits
    w = 1.0 / dist ** power
    return np.sum(w * z[idx], axis=1) / np.sum(w, axis=1)


def interpolate(xy, z, grid_xy, method):
    if method == "idw":
        return idw(xy, z, grid_xy, IDW_POWER, IDW_NEIGHBORS)
    if method == "rbf":
        rbf = RBFInterpolator(xy, z, kernel=RBF_KERNEL, neighbors=50, smoothing=0.0)
        return rbf(grid_xy)
    if method in ("linear", "cubic"):
        out = griddata(xy, z, grid_xy, method=method)
        # griddata leaves NaN outside the convex hull -> fill with nearest value
        nan = np.isnan(out)
        if nan.any():
            out[nan] = griddata(xy, z, grid_xy[nan], method="nearest")
        return out
    raise ValueError(f"Unknown METHOD: {method}")


def save_tif(path, arr, transform, crs):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
        nodata=NODATA, compress="lzw",
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def save_png(path, arr, extent, title, vmin, vmax, aoi):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.ma.masked_equal(arr, NODATA), extent=extent, cmap="YlGnBu",
                   vmin=vmin, vmax=vmax, origin="upper")
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="Rainfall (mm)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(PIXELS_CSV)
    year_cols = [c for c in df.columns if c.startswith("P_")]
    print(f"{len(df)} pixels, years: {[c[2:] for c in year_cols]}")

    aoi = gpd.read_file(AOI_SHP).to_crs(OUT_CRS)
    aoi_union = aoi.union_all()

    # Build output grid covering AOI (+ buffer)
    minx, miny, maxx, maxy = aoi.total_bounds
    minx -= BUFFER_DEG; miny -= BUFFER_DEG; maxx += BUFFER_DEG; maxy += BUFFER_DEG
    width  = int(np.ceil((maxx - minx) / CELL_SIZE))
    height = int(np.ceil((maxy - miny) / CELL_SIZE))
    transform = from_origin(minx, maxy, CELL_SIZE, CELL_SIZE)

    xs = minx + (np.arange(width)  + 0.5) * CELL_SIZE
    ys = maxy - (np.arange(height) + 0.5) * CELL_SIZE
    gx, gy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    # AOI mask (True = outside AOI)
    outside = geometry_mask([aoi_union], out_shape=(height, width),
                            transform=transform, invert=False)

    xy = df[["lon", "lat"]].to_numpy()

    # One common colour scale so years are visually comparable
    vmin = df[year_cols].min().min()
    vmax = df[year_cols].max().max()
    extent = (minx, maxx, miny, maxy)

    stack = []
    for col in year_cols:
        year = col[2:]
        z = df[col].to_numpy(dtype=float)
        surf = interpolate(xy, z, grid_xy, METHOD).reshape(height, width)
        stack.append(surf.copy())

        if CLIP_TO_AOI:
            surf[outside] = NODATA

        tif = OUT_DIR / f"rainfall_{year}_{METHOD}.tif"
        save_tif(tif, surf, transform, OUT_CRS)
        save_png(OUT_DIR / f"rainfall_{year}_{METHOD}.png", surf, extent,
                 f"Annual rainfall {year} (CHIRPS, {METHOD.upper()})", vmin, vmax, aoi)
        print(f"saved {tif}")

    # Multi-year mean raster
    mean_surf = np.mean(stack, axis=0)
    if CLIP_TO_AOI:
        mean_surf[outside] = NODATA
    first, last = year_cols[0][2:], year_cols[-1][2:]
    tif = OUT_DIR / f"rainfall_mean_{first}_{last}_{METHOD}.tif"
    save_tif(tif, mean_surf, transform, OUT_CRS)
    save_png(OUT_DIR / f"rainfall_mean_{first}_{last}_{METHOD}.png", mean_surf, extent,
             f"Mean annual rainfall {first}-{last} ({METHOD.upper()})", vmin, vmax, aoi)
    print(f"saved {tif}")

    # Optional: leave-one-out cross-validation so you can compare methods
    errs = []
    for col in year_cols:
        z = df[col].to_numpy(dtype=float)
        pred = np.empty_like(z)
        for i in range(len(z)):
            m = np.ones(len(z), bool); m[i] = False
            pred[i] = interpolate(xy[m], z[m], xy[i:i + 1], METHOD)[0]
        errs.append(np.sqrt(np.mean((pred - z) ** 2)))
    print(f"LOOCV RMSE ({METHOD}) per year [mm]:",
          dict(zip([c[2:] for c in year_cols], np.round(errs, 1))))


if __name__ == "__main__":
    main()
