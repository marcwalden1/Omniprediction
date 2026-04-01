"""Train OmniPredictor for each (variable, lead_time)."""
import logging
import sys
from pathlib import Path
import pickle
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_era5, load_ifs_ens, extract_quantiles
from data.normalization import Normalizer
from data.arches_runner import load_arches_quantiles
from tasks.frost_protection import FrostProtection
from tasks.heat_protection import HeatProtection
from tasks.wind_power import WindPowerDispatch
from omniprediction.algorithm import run_omniprediction


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["data"]["local_cache_dir"])
    output_dir = cache_dir / "omni_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    quantile_levels = cfg["omniprediction"]["quantile_levels"]
    epsilon = cfg["omniprediction"]["epsilon"]
    eta = cfg["omniprediction"]["eta"]
    max_iter = cfg["omniprediction"]["max_iterations"]
    var_names = [v["name"] for v in cfg["data"]["variables"]]
    lead_times = cfg["data"]["lead_times_hours"]
    arches_zarr = cfg["arches"]["local_zarr"]

    # Build tasks
    frost_cfg = cfg["tasks"]["frost_protection"]
    heat_cfg = cfg["tasks"]["heat_protection"]
    wind_cfg = cfg["tasks"]["wind_power"]

    tasks = [
        FrostProtection(
            theta_grid=frost_cfg["theta_grid"],
            c_ratio_grid=frost_cfg["c_ratio_grid"],
            scale=frost_cfg["scale"],
            transition_width=frost_cfg["transition_width"],
        ),
        HeatProtection(
            theta_grid=heat_cfg["theta_grid"],
            c_ratio_grid=heat_cfg["c_ratio_grid"],
            scale=heat_cfg["scale"],
            transition_width=heat_cfg["transition_width"],
        ),
        WindPowerDispatch(
            v_cutin=wind_cfg["v_cutin"],
            v_rated=wind_cfg["v_rated"],
            v_cutoff=wind_cfg["v_cutoff"],
            alpha_hellmann=wind_cfg["alpha_hellmann"],
            hub_height=wind_cfg["hub_height"],
            measurement_height=wind_cfg["measurement_height"],
            u_pen_grid=wind_cfg["u_pen_grid"],
        ),
    ]

    logger.info("Loading ERA5 and IFS ENS...")
    era5_ds = load_era5(
        gcs_path=cfg["data"]["era5_path"],
        variables=var_names,
        year=cfg["data"]["year"],
        region=cfg["data"]["region"],
        local_cache=cache_dir,
    )
    ifs_ds = load_ifs_ens(
        gcs_path=cfg["data"]["ifs_ens_path"],
        variables=var_names,
        year=cfg["data"]["year"],
        region=cfg["data"]["region"],
        lead_times_hours=lead_times,
        local_cache=cache_dir,
    )

    for var in var_names:
        for lt in lead_times:
            logger.info(f"Training OmniPredictor: var={var}, lead={lt}h")

            # Extract IFS ENS quantiles
            step_dim = "step" if "step" in ifs_ds.coords else "prediction_timedelta"
            ifs_lt = ifs_ds.sel({step_dim: np.timedelta64(lt, "h")})
            p_ifs = extract_quantiles(ifs_lt, var, quantile_levels)  # (N, d)

            # Extract ERA5 observations aligned to IFS
            init_times = ifs_lt.time.values
            valid_times = init_times + np.timedelta64(lt, "h")
            obs_da = era5_ds[var].sel(time=valid_times, method="nearest")
            y_obs = obs_da.values.reshape(-1)

            # Normalize
            normalizer = Normalizer.fit(y_obs, var)
            y_obs_norm = normalizer.normalize(y_obs)
            p_ifs_norm = normalizer.normalize(p_ifs)

            # Load Arches quantiles
            try:
                p_arches_norm = load_arches_quantiles(arches_zarr, var, quantile_levels)
                # Trim/pad to match N
                if p_arches_norm.shape[0] != p_ifs_norm.shape[0]:
                    n = min(p_arches_norm.shape[0], p_ifs_norm.shape[0])
                    p_arches_norm = p_arches_norm[:n]
                    p_ifs_norm = p_ifs_norm[:n]
                    y_obs_norm = y_obs_norm[:n]
            except Exception as e:
                logger.warning(f"Arches load failed ({e}), using IFS as Arches fallback.")
                p_arches_norm = p_ifs_norm.copy()

            # Only use tasks relevant to the variable
            if "wind" in var.lower() or "ws" in var.lower():
                var_tasks = [t for t in tasks if isinstance(t, WindPowerDispatch)]
            else:
                var_tasks = [t for t in tasks if not isinstance(t, WindPowerDispatch)]

            # Run algorithm
            p_final_norm, history = run_omniprediction(
                p_ifs=p_ifs_norm,
                p_arches=p_arches_norm,
                y_obs_norm=y_obs_norm,
                tasks=var_tasks,
                quantile_levels=quantile_levels,
                epsilon=epsilon,
                eta=eta,
                max_iterations=max_iter,
                verbose=True,
            )

            # Denormalize
            p_final = normalizer.denormalize(p_final_norm)

            # Save results
            result = {
                "variable": var,
                "lead_time_hours": lt,
                "p_omni": p_final,
                "p_ifs": normalizer.denormalize(p_ifs_norm),
                "y_obs": y_obs,
                "history": history,
                "normalizer": normalizer,
                "quantile_levels": quantile_levels,
            }
            out_file = output_dir / f"{var}_{lt}h.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(
                f"Saved to {out_file}. Converged: {history['converged']}, "
                f"updates: {history['n_updates']}"
            )


if __name__ == "__main__":
    main()
