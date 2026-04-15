"""Train OmniPredictor for each (variable, lead_time)."""
import logging
import os
import sys
from itertools import product as cartesian_product
from pathlib import Path
import pickle
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_era5, load_ifs_ens, extract_exceedance_probs
from data.normalization import Normalizer
from data.arches_runner import load_arches_exceedance_probs
from tasks.frost_protection import FrostProtection
from tasks.heat_protection import HeatProtection
from tasks.wind_power import WindPowerDispatch
from omniprediction.algorithm import run_omniprediction


def main():
    config_path = Path(
        os.environ.get(
            "OMNI_CONFIG",
            Path(__file__).parent.parent / "config" / "default.yaml",
        )
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["data"]["local_cache_dir"])
    output_dir = cache_dir / "omni_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    omni_cfg = cfg["omniprediction"]
    lam = float(omni_cfg["lam"])
    d = int(omni_cfg["d"])
    epsilon = float(omni_cfg["epsilon"])
    eta = float(omni_cfg["eta"])
    max_iter = int(omni_cfg["max_iterations"])

    tau = (np.arange(d) - d // 2) * lam  # (d,) normalized thresholds

    var_names = [v["name"] for v in cfg["data"]["variables"]]
    lead_times = cfg["data"]["lead_times_hours"]
    time_start = cfg["data"].get("time_start")
    time_stop = cfg["data"].get("time_stop")
    arches_zarr = cfg["arches"]["local_zarr"]

    # Build flat loss lists (one task instance per (task_type, param_combo))
    frost_cfg = cfg["tasks"]["frost_protection"]
    heat_cfg = cfg["tasks"]["heat_protection"]
    wind_cfg = cfg["tasks"]["wind_power"]

    frost_losses = [
        FrostProtection(theta, c, frost_cfg["scale"], frost_cfg["transition_width"], tau)
        for theta, c in cartesian_product(frost_cfg["theta_grid"], frost_cfg["c_ratio_grid"])
    ]
    heat_losses = [
        HeatProtection(theta, c, heat_cfg["scale"], heat_cfg["transition_width"], tau)
        for theta, c in cartesian_product(heat_cfg["theta_grid"], heat_cfg["c_ratio_grid"])
    ]
    wind_losses = [
        WindPowerDispatch(
            u_pen=u_pen,
            v_cutin=wind_cfg["v_cutin"],
            v_rated=wind_cfg["v_rated"],
            v_cutoff=wind_cfg["v_cutoff"],
            alpha_hellmann=wind_cfg["alpha_hellmann"],
            hub_height=wind_cfg["hub_height"],
            measurement_height=wind_cfg["measurement_height"],
            tau=tau,
        )
        for u_pen in wind_cfg["u_pen_grid"]
    ]

    logger.info("Loading ERA5 and IFS ENS...")
    era5_ds = load_era5(
        gcs_path=cfg["data"]["era5_path"],
        variables=var_names,
        year=cfg["data"]["year"],
        region=cfg["data"]["region"],
        time_start=time_start,
        time_stop=time_stop,
        local_cache=cache_dir,
    )
    ifs_ds = load_ifs_ens(
        gcs_path=cfg["data"]["ifs_ens_path"],
        variables=var_names,
        year=cfg["data"]["year"],
        region=cfg["data"]["region"],
        lead_times_hours=lead_times,
        time_start=time_start,
        time_stop=time_stop,
        local_cache=cache_dir,
    )

    for var in var_names:
        for lt in lead_times:
            logger.info("Training OmniPredictor: var=%s, lead=%dh", var, lt)

            step_dim = "step" if "step" in ifs_ds.coords else "prediction_timedelta"
            ifs_lt = ifs_ds.sel({step_dim: np.timedelta64(lt, "h")})

            # Align observations to IFS init times
            init_times = ifs_lt.time.values
            valid_times = init_times + np.timedelta64(lt, "h")
            obs_da = era5_ds[var].sel(time=valid_times, method="nearest")
            y_obs = obs_da.values.reshape(-1)

            # Fit normalizer on observations
            normalizer = Normalizer.fit(y_obs, var)
            y_obs_norm = normalizer.normalize(y_obs)

            # Extract IFS exceedance probs (normalized via same normalizer)
            p_ifs = extract_exceedance_probs(ifs_lt[var], tau, normalizer)  # (N, d)

            # Load Arches exceedance probs
            try:
                p_arches = load_arches_exceedance_probs(
                    arches_zarr, var, tau, lead_time_hours=lt, normalizer=normalizer
                )
                if p_arches.shape[0] != p_ifs.shape[0]:
                    raise ValueError(
                        f"Arches and IFS sample counts differ for {var} {lt}h: "
                        f"{p_arches.shape[0]} vs {p_ifs.shape[0]}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Arches exceedance probs for {var} {lt}h from "
                    f"{arches_zarr!r}: {e}. "
                    "Run scripts/run_arches.py first to generate the Arches forecast cache."
                ) from e

            # Select losses relevant to this variable
            if "wind" in var.lower() or "ws" in var.lower():
                var_losses = wind_losses
            else:
                var_losses = frost_losses + heat_losses

            # Run algorithm
            p_final, history = run_omniprediction(
                p_ifs=p_ifs,
                p_arches=p_arches,
                y_obs_norm=y_obs_norm,
                losses=var_losses,
                lam=lam,
                d=d,
                epsilon=epsilon,
                eta=eta,
                max_iterations=max_iter,
                verbose=True,
            )

            # Save results
            result = {
                "variable": var,
                "lead_time_hours": lt,
                "p_omni": p_final,
                "p_ifs": p_ifs,
                "y_obs": y_obs,
                "y_obs_norm": y_obs_norm,
                "history": history,
                "normalizer": normalizer,
                "lam": lam,
                "d": d,
                "tau": tau,
            }
            out_file = output_dir / f"{var}_{lt}h.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(
                "Saved to %s. Converged: %s, updates: %d",
                out_file,
                history["converged"],
                history["n_updates"],
            )


if __name__ == "__main__":
    main()
