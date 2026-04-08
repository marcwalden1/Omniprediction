"""Real ArchesWeatherGen inference and local forecast export."""
import importlib.resources
import logging
import os
from pathlib import Path
import shutil
import time

import certifi
import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from tensordict import TensorDict

from data.loader import _open_gcs_mapper

logger = logging.getLogger(__name__)

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SURFACE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]
_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
]
_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
_GLOBAL_LATS = np.arange(90, -90.0 - 1e-6, -1.5, dtype=np.float32)
_GLOBAL_LONS = np.arange(-180, 180, 1.5, dtype=np.float32)


def _target_region_coords(region: dict) -> tuple[np.ndarray, np.ndarray]:
    resolution = float(region.get("resolution", 1.5))
    lats = np.arange(region["lat_max"], region["lat_min"] - 1e-6, -resolution, dtype=np.float32)
    lons = np.arange(region["lon_min"], region["lon_max"] + 1e-6, resolution, dtype=np.float32)
    return lats, lons


def _normalize_longitudes(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    normalized = (((ds[lon_name] + 180.0) % 360.0) - 180.0).astype(np.float32)
    return ds.assign_coords({lon_name: normalized}).sortby(lon_name)


def _regrid_global_1p5(ds: xr.Dataset) -> xr.Dataset:
    ds = _normalize_longitudes(ds)
    try:
        return ds.sel(latitude=_GLOBAL_LATS, longitude=_GLOBAL_LONS, method="nearest")
    except Exception:
        return ds.interp(latitude=_GLOBAL_LATS, longitude=_GLOBAL_LONS)


def _is_valid_arches_store(path: Path) -> bool:
    try:
        ds = xr.open_zarr(path)
    except Exception:
        return False
    required_dims = {"time", "member"}
    has_data = bool(ds.data_vars)
    return has_data and required_dims.issubset(ds.dims)


def _load_geoarches_stats():
    import geoarches.stats as geoarches_stats

    stats_root = importlib.resources.files(geoarches_stats)
    pangu_stats = torch.load(stats_root / "pangu_norm_stats2_with_w.pt", weights_only=True)
    return TensorDict(
        surface=pangu_stats["surface_mean"],
        level=pangu_stats["level_mean"],
    ), TensorDict(
        surface=pangu_stats["surface_std"],
        level=pangu_stats["level_std"],
    )


def _load_real_arches_module():
    from geoarches.lightning_modules import load_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module, cfg = load_module("archesweathergen", device=device)
    return module, cfg, device


def _load_era5_init_states(
    init_times: np.ndarray,
    time_start: str | None = None,
    time_stop: str | None = None,
) -> xr.Dataset:
    mapper = _open_gcs_mapper(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    )
    ds = xr.open_zarr(mapper, consolidated=True)
    needed_times = pd.to_datetime(init_times)
    prev_times = needed_times - pd.Timedelta(hours=24)
    all_times = pd.Index(sorted(set(needed_times.tolist() + prev_times.tolist())))
    ds = ds[_SURFACE_VARS + _LEVEL_VARS]
    ds = ds.sel(level=_PRESSURE_LEVELS)
    if time_start or time_stop:
        ds = ds.sel(
            time=slice(
                pd.Timestamp(time_start) - pd.Timedelta(hours=24) if time_start else None,
                pd.Timestamp(time_stop) + pd.Timedelta(days=1) if time_stop else None,
            )
        )
    ds = ds.sel(time=all_times, method="nearest")
    return _regrid_global_1p5(ds)


class _GeoArchesAdapter:
    def __init__(self):
        self.means, self.stds = _load_geoarches_stats()

    def _to_tensordict(self, ds_slice: xr.Dataset) -> TensorDict:
        ds_slice = ds_slice.sel(level=_PRESSURE_LEVELS)
        ds_slice = ds_slice.transpose(..., "level", "latitude", "longitude")
        ds_slice = _normalize_longitudes(ds_slice)

        surface = ds_slice[_SURFACE_VARS].to_array().to_numpy().astype(np.float32, copy=False)
        level = ds_slice[_LEVEL_VARS].to_array().to_numpy().astype(np.float32, copy=False)

        surface_t = torch.from_numpy(surface).unsqueeze(-3)
        level_t = torch.from_numpy(level)

        half_lon = surface_t.shape[-1] // 2
        surface_t = surface_t.roll(half_lon, -1)
        level_t = level_t.roll(half_lon, -1)

        return TensorDict({"surface": surface_t, "level": level_t}, batch_size=[])

    def build_batch(self, era5_states: xr.Dataset, init_times: np.ndarray) -> dict[str, torch.Tensor]:
        states = []
        prev_states = []
        timestamps = []

        for raw_time in pd.to_datetime(init_times):
            state = self._to_tensordict(era5_states.sel(time=raw_time))
            prev_state = self._to_tensordict(era5_states.sel(time=raw_time - pd.Timedelta(hours=24)))
            states.append(state)
            prev_states.append(prev_state)
            timestamps.append(int(raw_time.value // 10**9))

        state_batch = torch.stack(states, dim=0)
        prev_batch = torch.stack(prev_states, dim=0)

        means = self.means.to("cpu")
        stds = self.stds.to("cpu")
        state_batch = (state_batch - means) / stds
        prev_batch = (prev_batch - means) / stds

        return {
            "timestamp": torch.tensor(timestamps, dtype=torch.int32),
            "lead_time_hours": torch.full((len(timestamps),), 24, dtype=torch.int32),
            "state": state_batch,
            "prev_state": prev_batch,
        }

    def denormalize(self, state_batch: TensorDict) -> TensorDict:
        means = self.means.to(state_batch.device)
        stds = self.stds.to(state_batch.device)
        return state_batch * stds + means

    def to_xarray(self, state_batch: TensorDict, timestamps: torch.Tensor) -> xr.Dataset:
        state_batch = state_batch.cpu()
        half_lon = state_batch["surface"].shape[-1] // 2
        rolled = state_batch.apply(lambda x: x.roll(-half_lon, -1))
        surface = rolled["surface"].squeeze(-3).numpy()
        level = rolled["level"].numpy()
        times = pd.to_datetime(timestamps.cpu().numpy(), unit="s").tz_localize(None)

        ds = xr.Dataset(
            data_vars={
                **{
                    name: (["time", "level", "latitude", "longitude"], level[:, idx])
                    for idx, name in enumerate(_LEVEL_VARS)
                },
                **{
                    name: (["time", "latitude", "longitude"], surface[:, idx])
                    for idx, name in enumerate(_SURFACE_VARS)
                },
            },
            coords={
                "time": times,
                "level": _PRESSURE_LEVELS,
                "latitude": _GLOBAL_LATS,
                "longitude": _GLOBAL_LONS,
            },
        )
        return ds


def run_arches_inference(
    ifs_ds,
    variables: list[str],
    lead_times_hours: list[int],
    n_samples: int,
    output_zarr: str,
    model_name: str = "openclimatefix/ArchesWeatherGen",
    region: dict | None = None,
    time_start: str | None = None,
    time_stop: str | None = None,
):
    """
    Generate forecast members with the real geoarches model when available.

    Forecasts are exported on the configured regional 1.5° grid with dimensions:
    (time, prediction_timedelta, latitude, longitude, member).
    """
    del model_name
    output_path = Path(output_zarr)
    if output_path.exists():
        if _is_valid_arches_store(output_path):
            logger.info(f"Arches cache exists at {output_zarr}, skipping inference.")
            return zarr.open(output_zarr, mode="r")
        broken_path = output_path.with_name(f"{output_path.name}.broken.{int(time.time())}")
        logger.warning("Existing Arches cache at %s is invalid; moving it to %s", output_zarr, broken_path)
        shutil.move(output_path, broken_path)

    try:
        _run_real_arches(
            ifs_ds=ifs_ds,
            variables=variables,
            lead_times_hours=lead_times_hours,
            n_samples=n_samples,
            output_zarr=output_zarr,
            region=region,
            time_start=time_start,
            time_stop=time_stop,
        )
    except Exception as exc:
        logger.warning("Real Arches inference failed (%s); using Gaussian fallback.", exc)
        _run_fallback_arches(ifs_ds, variables, lead_times_hours, n_samples, output_zarr)

    return zarr.open(output_zarr, mode="r")


def _run_real_arches(
    ifs_ds,
    variables: list[str],
    lead_times_hours: list[int],
    n_samples: int,
    output_zarr: str,
    region: dict | None,
    time_start: str | None,
    time_stop: str | None,
):
    import geoarches  # noqa: F401

    if region is None:
        raise ValueError("region is required for real Arches inference")
    if any(lead % 24 for lead in lead_times_hours):
        raise ValueError("Arches integration only supports lead times that are multiples of 24h")

    output_path = Path(output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, _, device = _load_real_arches_module()
    adapter = _GeoArchesAdapter()

    init_times = pd.to_datetime(ifs_ds.time.values)
    era5_states = _load_era5_init_states(init_times.values, time_start=time_start, time_stop=time_stop)

    max_rollout_steps = max(lead_times_hours) // 24
    region_lats, region_lons = _target_region_coords(region)
    batch_size = int(os.environ.get("OMNI_ARCHES_BATCH_SIZE", "1"))
    lead_timedeltas = pd.to_timedelta(lead_times_hours, unit="h")
    first_batch = True

    for batch_idx, start in enumerate(range(0, len(init_times), batch_size)):
        batch_times = init_times[start : start + batch_size]
        logger.info(
            "Running real Arches batch %s/%s for init times %s -> %s",
            batch_idx + 1,
            int(np.ceil(len(init_times) / batch_size)),
            batch_times[0],
            batch_times[-1],
        )
        batch = adapter.build_batch(era5_states, batch_times.values)
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

        member_rollouts = []
        for member in range(n_samples):
            logger.info(
                "Sampling real Arches member %s/%s for batch %s",
                member + 1,
                n_samples,
                batch_idx + 1,
            )
            rollout = model.sample_rollout(
                batch,
                batch_nb=batch_idx,
                member=member,
                iterations=max_rollout_steps,
                disable_tqdm=True,
            )
            member_rollouts.append(adapter.denormalize(rollout).cpu())

        step_datasets = []
        for step in range(max_rollout_steps):
            members = []
            step_timestamp = batch["timestamp"].cpu()
            for rollout in member_rollouts:
                members.append(adapter.to_xarray(rollout[:, step], step_timestamp))
            step_ds = xr.concat(members, dim=pd.Index(range(n_samples), name="member"))
            step_ds = _normalize_longitudes(step_ds).sel(
                latitude=region_lats,
                longitude=region_lons,
                method="nearest",
            )
            step_ds["10m_wind_speed"] = np.hypot(
                step_ds["10m_u_component_of_wind"],
                step_ds["10m_v_component_of_wind"],
            )
            export_vars = ["2m_temperature"]
            if "10m_wind_speed" in variables:
                export_vars.append("10m_wind_speed")
            step_ds = step_ds[export_vars]
            step_datasets.append(step_ds)

        export = xr.concat(
            step_datasets,
            dim=pd.Index(lead_timedeltas[:max_rollout_steps], name="prediction_timedelta"),
        )
        export = export.transpose("time", "prediction_timedelta", "latitude", "longitude", "member")

        if first_batch:
            export.to_zarr(output_zarr, mode="w")
            first_batch = False
        else:
            export.to_zarr(output_zarr, append_dim="time")
        logger.info("Wrote real Arches batch %s to %s", batch_idx + 1, output_zarr)

    logger.info("Arches forecasts saved to %s", output_zarr)


def _run_fallback_arches(ifs_ds, variables, lead_times_hours, n_samples, output_zarr):
    """Fallback: perturb IFS ENS mean with scaled noise."""
    logger.info("Generating Arches fallback samples via Gaussian perturbation.")

    output_path = Path(output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step_dim = "step" if "step" in ifs_ds.coords else "prediction_timedelta"
    coords = {
        "time": ifs_ds["time"],
        step_dim: ifs_ds[step_dim],
        "latitude": ifs_ds["latitude"],
        "longitude": ifs_ds["longitude"],
        "member": np.arange(n_samples, dtype=np.int32),
    }
    arrays = {}

    for var in variables:
        da = ifs_ds[var]
        member_dim = next(
            (d for d in da.dims if any(k in d for k in ("member", "number", "realization"))),
            None,
        )
        if member_dim:
            mean = da.mean(dim=member_dim)
            std = da.std(dim=member_dim)
        else:
            mean = da
            std = da * 0.1

        mean_vals = mean.values
        std_vals = std.values
        rng = np.random.default_rng(42)
        samples = (
            mean_vals[..., np.newaxis]
            + rng.normal(0, 1, (*mean_vals.shape, n_samples)) * std_vals[..., np.newaxis]
        )
        arrays[var] = (list(mean.dims) + ["member"], samples.astype(np.float32, copy=False))

    xr.Dataset(data_vars=arrays, coords=coords).to_zarr(output_zarr, mode="w")

    logger.info("Arches fallback saved to %s", output_zarr)


def load_arches_quantiles(
    zarr_path: str,
    variable: str,
    quantile_levels: list[float],
    lead_time_hours: int,
) -> np.ndarray:
    """Load lead-specific Arches samples and compute quantiles → (N, d)."""
    ds = xr.open_zarr(zarr_path)
    step_dim = "step" if "step" in ds.coords else "prediction_timedelta"
    da = ds[variable].sel({step_dim: np.timedelta64(lead_time_hours, "h")})
    samples = da.values
    qs = np.quantile(samples, quantile_levels, axis=-1)
    return qs.reshape(len(quantile_levels), -1).T
