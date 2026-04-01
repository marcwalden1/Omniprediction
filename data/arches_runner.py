"""ArchesWeatherGen inference — run once, cache to local zarr."""
import logging
from pathlib import Path
import numpy as np
import zarr

logger = logging.getLogger(__name__)


def run_arches_inference(
    ifs_ds,
    variables: list[str],
    lead_times_hours: list[int],
    n_samples: int,
    output_zarr: str,
    model_name: str = "openclimatefix/ArchesWeatherGen",
):
    """
    Run ArchesWeatherGen to generate ensemble samples.

    Falls back to a Gaussian perturbation of IFS ENS mean if the
    HuggingFace model is unavailable (for testing/CI).
    """
    output_path = Path(output_zarr)
    if output_path.exists():
        logger.info(f"Arches cache exists at {output_zarr}, skipping inference.")
        return zarr.open(output_zarr, mode="r")

    try:
        from archesweathergen import ArchesWeatherGen  # type: ignore
        _run_real_arches(ifs_ds, variables, lead_times_hours, n_samples, output_zarr, model_name)
    except ImportError:
        logger.warning(
            "ArchesWeatherGen not installed — using Gaussian perturbation fallback."
        )
        _run_fallback_arches(ifs_ds, variables, lead_times_hours, n_samples, output_zarr)

    return zarr.open(output_zarr, mode="r")


def _run_fallback_arches(ifs_ds, variables, lead_times_hours, n_samples, output_zarr):
    """Fallback: perturb IFS ENS mean with scaled noise."""
    import xarray as xr
    logger.info("Generating Arches fallback samples via Gaussian perturbation.")

    # Ensure parent directory exists
    output_path = Path(output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(output_zarr, mode="w")
    step_dim = "step" if "step" in ifs_ds.coords else "prediction_timedelta"

    for var in variables:
        da = ifs_ds[var]
        member_dim = next(
            (d for d in da.dims if any(k in d for k in ("member", "number", "realization"))),
            None,
        )
        # Compute ensemble mean
        if member_dim:
            mean = da.mean(dim=member_dim)
            std = da.std(dim=member_dim)
        else:
            mean = da
            std = da * 0.1

        mean_vals = mean.values  # (time, step_or_lead, lat, lon) or similar
        std_vals = std.values
        # Generate n_samples perturbations
        rng = np.random.default_rng(42)
        samples = (
            mean_vals[..., np.newaxis]
            + rng.normal(0, 1, (*mean_vals.shape, n_samples)) * std_vals[..., np.newaxis]
        )
        store[var] = samples
        store[var].attrs["dimensions"] = list(mean.dims) + ["member"]
        store[var].attrs["lead_times_hours"] = lead_times_hours

    logger.info(f"Arches fallback saved to {output_zarr}")


def _run_real_arches(ifs_ds, variables, lead_times_hours, n_samples, output_zarr, model_name):
    """Real ArchesWeatherGen inference via HuggingFace."""
    from archesweathergen import ArchesWeatherGen  # type: ignore
    import torch

    logger.info(f"Loading {model_name} from HuggingFace...")
    model = ArchesWeatherGen.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Ensure parent directory exists
    output_path = Path(output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(output_zarr, mode="w")
    with torch.no_grad():
        for var in variables:
            samples_list = []
            for _ in range(n_samples):
                pred = model.predict(ifs_ds, variable=var, lead_times=lead_times_hours)
                samples_list.append(pred.values)
            samples = np.stack(samples_list, axis=-1)
            store[var] = samples
    logger.info(f"Arches forecasts saved to {output_zarr}")


def load_arches_quantiles(
    zarr_path: str,
    variable: str,
    quantile_levels: list[float],
) -> np.ndarray:
    """Load Arches zarr and compute quantiles → (N, d)."""
    store = zarr.open(zarr_path, mode="r")
    samples = store[variable][:]  # (..., n_members)
    d = len(quantile_levels)
    qs = np.quantile(samples, quantile_levels, axis=-1)  # (d, ...)
    return qs.reshape(d, -1).T  # (N, d)
