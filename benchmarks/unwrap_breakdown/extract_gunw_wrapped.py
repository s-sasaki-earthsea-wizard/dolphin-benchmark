#!/usr/bin/env python
"""Extract the wrapped interferogram + coherence from a NISAR L2 GUNW product.

NISAR GUNW products carry a real wrapped interferogram at the fine (20 m)
grid alongside the official 80 m unwrapped product::

    //science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH/
        wrappedInterferogram   (CFloat32, e.g. 17604 x 17352)
        coherenceMagnitude     (Float32, same grid)

Multilooking the complex interferogram (and coherence) over k x k blocks
(default 4 -> the 80 m grid) reproduces the mission's multilook-then-unwrap
configuration and yields realistic unwrap inputs at frame scale.

Outputs follow dolphin's stitched-interferogram naming
(``<ref>_<sec>.int.tif`` + ``<ref>_<sec>.int.cor.tif``) so
``profile_unwrap.py --ifg-files`` can consume them directly.

Invalid pixels (NaN or exactly-zero complex fill) are excluded from the
block average; fully-invalid blocks become 0 (ifg) / 0.0 (coherence), the
same convention dolphin's stitched rasters use for missing data.

Effective number of looks: the GUNW carries no numberOfLooks metadata for
the wrapped layer, so pass ``--nlooks k*k`` (16 for the default 4x4) to the
profiler — a lower bound, since the 20 m layer itself is already
multilooked. SNAPHU uses nlooks only for cost weighting; timing conclusions
are insensitive to this.

Usage (inside the dev container, with the GUNW dir mounted)::

    python extract_gunw_wrapped.py --gunw /gunw/NISAR_L2_PR_GUNW_..._001.h5 \
        --out /work/nisar-unwrap-bench -k 4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

GRIDS = "science/LSAR/GUNW/grids/frequencyA"
WRAP_GROUP = f"{GRIDS}/wrappedInterferogram"
ROW_CHUNK = 2048  # rows per read; keeps peak memory ~small multiples of a chunk

TIFF_OPTIONS = ["COMPRESS=DEFLATE", "ZLEVEL=4", "TILED=YES", "PREDICTOR=1"]


def pair_name(gunw_path: Path) -> str:
    """``..._<refstart>_<refend>_<secstart>_<secend>_...`` -> ``<ref>_<sec>``."""
    stamps = re.findall(r"_(\d{8})T\d{6}", gunw_path.name)
    if len(stamps) < 3:
        raise ValueError(f"cannot parse acquisition dates from {gunw_path.name}")
    return f"{stamps[0]}_{stamps[2]}"


def multilook(arr: np.ndarray, k: int, out_dtype) -> np.ndarray:
    """k x k block average, excluding NaN and exact-zero fill."""
    ny, nx = arr.shape
    ny2, nx2 = ny // k, nx // k
    a = arr[: ny2 * k, : nx2 * k].reshape(ny2, k, nx2, k)
    valid = np.isfinite(a) & (a != 0)
    total = np.where(valid, a, 0).sum(axis=(1, 3))
    count = valid.sum(axis=(1, 3))
    out = total / np.maximum(count, 1)
    out[count == 0] = 0
    return out.astype(out_dtype)


def multilook_chunked(dset: h5py.Dataset, k: int, out_dtype) -> np.ndarray:
    ny, nx = dset.shape
    ny2 = ny // k
    out = np.empty((ny2, nx // k), dtype=out_dtype)
    step = (ROW_CHUNK // k) * k
    for r0 in range(0, ny2 * k, step):
        r1 = min(r0 + step, ny2 * k)
        out[r0 // k : r1 // k] = multilook(dset[r0:r1], k, out_dtype)
    return out


def read_georef(h5: h5py.File, k: int):
    """Build (geotransform, wkt) for the multilooked grid, or (None, None)."""
    group = h5[WRAP_GROUP]
    xc = group.get("xCoordinates") or h5[GRIDS].get("xCoordinates")
    yc = group.get("yCoordinates") or h5[GRIDS].get("yCoordinates")
    proj = group.get("projection") or h5[GRIDS].get("projection")
    if xc is None or yc is None:
        return None, None
    x = xc[:]
    y = yc[:]
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    # Coordinates are pixel centers; shift half a pixel, then scale by k.
    gt = (float(x[0]) - dx / 2, dx * k, 0.0, float(y[0]) - dy / 2, 0.0, dy * k)
    wkt = None
    if proj is not None:
        epsg = proj.attrs.get("epsg_code")
        if epsg is not None:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(epsg))
            wkt = srs.ExportToWkt()
    return gt, wkt


def write_gtiff(path: Path, arr: np.ndarray, gdal_type, geotransform, wkt) -> None:
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path), arr.shape[1], arr.shape[0], 1, gdal_type, options=TIFF_OPTIONS
    )
    if geotransform is not None:
        ds.SetGeoTransform(geotransform)
    if wkt:
        ds.SetProjection(wkt)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract + multilook the wrapped igram from a NISAR GUNW"
    )
    p.add_argument("--gunw", required=True, nargs="+", help="GUNW HDF5 file(s)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("-k", "--looks", type=int, default=4, help="multilook factor")
    p.add_argument("--pol", default="HH")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gunw in map(Path, args.gunw):
        name = pair_name(gunw)
        ifg_out = out_dir / f"{name}.int.tif"
        cor_out = out_dir / f"{name}.int.cor.tif"
        if ifg_out.exists() and cor_out.exists():
            print(f"{name}: already extracted, skipping")
            continue
        with h5py.File(gunw, "r") as h5:
            grp = h5[f"{WRAP_GROUP}/{args.pol}"]
            print(f"{name}: multilooking {grp['wrappedInterferogram'].shape} "
                  f"by {args.looks}x{args.looks} ...")
            ifg = multilook_chunked(grp["wrappedInterferogram"], args.looks, np.complex64)
            cor = multilook_chunked(grp["coherenceMagnitude"], args.looks, np.float32)
            gt, wkt = read_georef(h5, args.looks)
        write_gtiff(ifg_out, ifg, gdal.GDT_CFloat32, gt, wkt)
        write_gtiff(cor_out, cor, gdal.GDT_Float32, gt, wkt)
        print(f"{name}: wrote {ifg_out.name} + {cor_out.name} shape={ifg.shape}")


if __name__ == "__main__":
    main()
