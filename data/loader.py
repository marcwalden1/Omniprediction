"""WeatherBench2 GCS data loading with local caching."""
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import zarr
import gcsfs

logger = logging.getLogger(__name__)


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
    local_cache: Optional[Path] = None,
) -> xr.Dataset:
    """Load ERA5 ground truth from GCS or local cache."""
    cache_path = local_cache / f"era5_{year}.zarr" if local_cache else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading ERA5 from cache: {cache_path}")
        return xr.open_zarr(str(cache_path))

    # Prepend bucket name if not already a full gs:// path
    full_path = gcs_path if gcs_path.startswith("gs://") else f"gs://weatherbench2/{gcs_path}"
    logger.info(f"Streaming ERA5 from GCS: {full_path}")
    fs = gcsfs.GCSFileSystem(token="google_default")
    mapper = fs.get_mapper(full_path)
    ds = xr.open_zarr(mapper, consolidated=True)

    # Select year
    ds = ds.sel(time=str(year))

    # Select region — handle both N→S and S→N latitude ordering
    lat = ds.latitude.values
    if lat[0] > lat[-1]:
        region_sel = dict(
            latitude=slice(region["lat_max"], region["lat_min"]),
            longitude=slice(region["lon_min"], region["lon_max"]),
        )
    else:
        region_sel = dict(
            latitude=slice(region["lat_min"], region["lat_max"]),
            longitude=slice(region["lon_min"], region["lon_max"]),
        )
    ds = ds.sel(**region_sel)

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
    ds = ds[keep]

    if cache_path:
        logger.info(f"Caching ERA5 to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(cache_path))

    return ds


def load_ifs_ens(
    gcs_path: str,
    variables: list[str],
    year: int,
    region: dict,
    lead_times_hours: list[int],
    local_cache: Optional[Path] = None,
) -> xr.Dataset:
    """Load IFS ENS forecasts from GCS or local cache."""
    cache_path = local_cache / f"ifs_ens_{year}.zarr" if local_cache else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading IFS ENS from cache: {cache_path}")
        return xr.open_zarr(str(cache_path))

    full_path = gcs_path if gcs_path.startswith("gs://") else f"gs://weatherbench2/{gcs_path}"
    logger.info(f"Streaming IFS ENS from GCS: {full_path}")
    fs = gcsfs.GCSFileSystem(token="google_default")
    mapper = fs.get_mapper(full_path)
    ds = xr.open_zarr(mapper, consolidated=True)

    # Select year
    init_times = ds.time.values
    mask = np.array([str(t)[:4] == str(year) for t in init_times])
    ds = ds.isel(time=np.where(mask)[0])

    # Select lead times
    if "step" in ds.coords:
        steps = [np.timedelta64(h, "h") for h in lead_times_hours]
        ds = ds.sel(step=steps)
    elif "prediction_timedelta" in ds.coords:
        steps = [np.timedelta64(h, "h") for h in lead_times_hours]
        ds = ds.sel(prediction_timedelta=steps)

    # Select region
    lat = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    if lat[0] > lat[-1]:
        region_sel = dict(
            latitude=slice(region["lat_max"], region["lat_min"]),
            longitude=slice(region["lon_min"], region["lon_max"]),
        )
    else:
        region_sel = dict(
            latitude=slice(region["lat_min"], region["lat_max"]),
            longitude=slice(region["lon_min"], region["lon_max"]),
        )
    ds = ds.sel(**region_sel)

    if cache_path:
        logger.info(f"Caching IFS ENS to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(cache_path))

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
