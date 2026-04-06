"""WeatherBench2 GCS data loading with local caching."""
import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import zarr
import gcsfs
import certifi
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _open_gcs_mapper(full_path: str):
    """Open a public GCS path, preferring ADC when available and falling back to anon."""
    try:
        fs = gcsfs.GCSFileSystem(token="google_default")
    except DefaultCredentialsError:
        logger.info("Google ADC not found; retrying GCS access anonymously.")
        fs = gcsfs.GCSFileSystem(token="anon")
    return fs.get_mapper(full_path)


def _maybe_disable_cache(local_cache: Optional[Path]) -> Optional[Path]:
    """Allow smoke runs to bypass local cache writes and reads."""
    if os.environ.get("OMNI_DISABLE_CACHE") == "1":
        return None
    return local_cache


def _normalize_longitude_coord(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lon_name not in ds.coords:
        return ds
    normalized = (((ds[lon_name] + 180.0) % 360.0) - 180.0).astype(np.float32)
    return ds.assign_coords({lon_name: normalized}).sortby(lon_name)


def _target_region_coords(region: dict) -> tuple[np.ndarray, np.ndarray]:
    resolution = float(region.get("resolution", 0.25))
    lats = np.arange(region["lat_max"], region["lat_min"] - 1e-6, -resolution, dtype=np.float32)
    lons = np.arange(region["lon_min"], region["lon_max"] + 1e-6, resolution, dtype=np.float32)
    return lats, lons


def _cache_suffix(region: dict, time_start: Optional[str], time_stop: Optional[str]) -> str:
    resolution = str(region.get("resolution", 0.25)).replace(".", "p")
    start = (time_start or "full").replace(":", "").replace("-", "")
    stop = (time_stop or "full").replace(":", "").replace("-", "")
    return f"r{resolution}_{start}_{stop}"


def _select_region(ds: xr.Dataset, region: dict) -> xr.Dataset:
    ds = _normalize_longitude_coord(ds)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lat = ds[lat_name].values
    if lat[0] > lat[-1]:
        ds = ds.sel(
            {
                lat_name: slice(region["lat_max"], region["lat_min"]),
                lon_name: slice(region["lon_min"], region["lon_max"]),
            }
        )
    else:
        ds = ds.sel(
            {
                lat_name: slice(region["lat_min"], region["lat_max"]),
                lon_name: slice(region["lon_min"], region["lon_max"]),
            }
        )
    target_lats, target_lons = _target_region_coords(region)
    try:
        ds = ds.sel({lat_name: target_lats, lon_name: target_lons}, method="nearest")
    except Exception:
        ds = ds.interp({lat_name: target_lats, lon_name: target_lons})
    return ds


def get_europe_slice(lat_min=35.0, lat_max=72.0, lon_min=-12.0, lon_max=42.0):
    return dict(
        latitude=slice(lat_max, lat_min),  # ERA5 is N→S
        longitude=slice(lon_min, lon_max),
    )


def load_era5(
    gcs_path: str,
    variables: list[str],
    year: int,
    region: dict,
    time_start: Optional[str] = None,
    time_stop: Optional[str] = None,
    local_cache: Optional[Path] = None,
) -> xr.Dataset:
    """Load ERA5 ground truth from GCS or local cache."""
    local_cache = _maybe_disable_cache(local_cache)
    suffix = _cache_suffix(region, time_start, time_stop)
    cache_path = local_cache / f"era5_{year}_{suffix}_zarr2.zarr" if local_cache else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading ERA5 from cache: {cache_path}")
        return xr.open_zarr(str(cache_path))

    # Prepend bucket name if not already a full gs:// path
    full_path = gcs_path if gcs_path.startswith("gs://") else f"gs://weatherbench2/{gcs_path}"
    logger.info(f"Streaming ERA5 from GCS: {full_path}")
    mapper = _open_gcs_mapper(full_path)
    ds = xr.open_zarr(mapper, consolidated=True)

    # Select year
    ds = ds.sel(time=str(year))
    if time_start or time_stop:
        ds = ds.sel(time=slice(time_start or None, time_stop or None))

    # Keep only requested variables (map friendly names)
    var_map = {
        "2m_temperature": ["2m_temperature", "t2m", "T2M"],
        "10m_wind_speed": ["10m_wind_speed", "ws10", "wind_speed"],
        "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10", "U10M"],
        "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10", "V10M"],
    }
    available = list(ds.data_vars)
    keep = []
    for v in variables:
        for alias in var_map.get(v, [v]):
            if alias in available:
                keep.append(alias)
                break
    ds = _select_region(ds[keep], region)

    if cache_path:
        logger.info(f"Caching ERA5 to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(cache_path), zarr_version=2)

    return ds


def load_ifs_ens(
    gcs_path: str,
    variables: list[str],
    year: int,
    region: dict,
    lead_times_hours: list[int],
    time_start: Optional[str] = None,
    time_stop: Optional[str] = None,
    local_cache: Optional[Path] = None,
) -> xr.Dataset:
    """Load IFS ENS forecasts from GCS or local cache."""
    local_cache = _maybe_disable_cache(local_cache)
    suffix = _cache_suffix(region, time_start, time_stop)
    cache_path = local_cache / f"ifs_ens_{year}_{suffix}_zarr2.zarr" if local_cache else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading IFS ENS from cache: {cache_path}")
        return xr.open_zarr(str(cache_path))

    full_path = gcs_path if gcs_path.startswith("gs://") else f"gs://weatherbench2/{gcs_path}"
    logger.info(f"Streaming IFS ENS from GCS: {full_path}")
    mapper = _open_gcs_mapper(full_path)
    ds = xr.open_zarr(mapper, consolidated=True)

    # Select year
    init_times = ds.time.values
    mask = np.array([str(t)[:4] == str(year) for t in init_times])
    ds = ds.isel(time=np.where(mask)[0])
    if time_start or time_stop:
        ds = ds.sel(time=slice(time_start or None, time_stop or None))

    # Select lead times
    if "step" in ds.coords:
        steps = [np.timedelta64(h, "h") for h in lead_times_hours]
        ds = ds.sel(step=steps)
    elif "prediction_timedelta" in ds.coords:
        steps = [np.timedelta64(h, "h") for h in lead_times_hours]
        ds = ds.sel(prediction_timedelta=steps)

    ds = _select_region(ds[variables], region)

    if cache_path:
        logger.info(f"Caching IFS ENS to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(cache_path), zarr_version=2)

    return ds


def extract_quantiles(
    ens_ds: xr.Dataset,
    variable: str,
    quantile_levels: list[float],
) -> np.ndarray:
    """
    Extract empirical quantiles from ensemble dimension.

    Returns array of shape (N, d) where N = n_samples (flattened grid×time)
    and d = len(quantile_levels).
    """
    da = ens_ds[variable]
    # Find member dimension
    member_dim = None
    for dim in da.dims:
        if "member" in dim or "number" in dim or "realization" in dim:
            member_dim = dim
            break
    if member_dim is None:
        raise ValueError(f"No member dimension found in {da.dims}")

    qs = da.quantile(quantile_levels, dim=member_dim).values
    # qs shape: (d, ...) → reshape to (N, d)
    d = len(quantile_levels)
    qs = qs.reshape(d, -1).T  # (N, d)
    return qs


def align_obs_and_forecasts(
    era5_ds: xr.Dataset,
    ifs_ds: xr.Dataset,
    variable: str,
    lead_time_hours: int,
) -> tuple[np.ndarray, xr.DataArray]:
    """
    Align ERA5 observations with IFS ENS forecasts for a given lead time.

    Returns:
        y_obs: (N,) observed values
        ens_da: DataArray with member dim, shape (N_time, N_lat, N_lon, N_members)
    """
    # Determine valid time from init_time + lead_time
    step = np.timedelta64(lead_time_hours, "h")
    step_dim = "step" if "step" in ifs_ds.coords else "prediction_timedelta"

    ens_da = ifs_ds[variable].sel({step_dim: step})
    init_times = ens_da.time.values
    valid_times = init_times + step

    # Select ERA5 at valid times
    obs_da = era5_ds[variable].sel(time=valid_times, method="nearest")

    # Flatten spatial dims for both
    member_dim = None
    for dim in ens_da.dims:
        if "member" in dim or "number" in dim or "realization" in dim:
            member_dim = dim
            break

    # Stack to (N, members)
    spatial_dims = [d for d in ens_da.dims if d not in ("time", member_dim, "step", step_dim)]
    y_obs = obs_da.values.reshape(-1)
    return y_obs, ens_da
