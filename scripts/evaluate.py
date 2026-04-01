"""Produce all metrics and figures."""
import logging
import sys
from pathlib import Path
import pickle
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.proper_scores import mean_crps, pit_histogram, sharpness_spread_ratio
from evaluation.decision_costs import compare_models
from evaluation.ranking import rank_models, cross_task_ranking_stability
from tasks.frost_protection import FrostProtection
from tasks.heat_protection import HeatProtection
from tasks.wind_power import WindPowerDispatch
from plots.figures import plot_ranking_heatmap, plot_convergence, plot_crps_comparison


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["data"]["local_cache_dir"])
    results_dir = cache_dir / "omni_results"
    fig_dir = cache_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    quantile_levels = np.array(cfg["omniprediction"]["quantile_levels"])
    var_names = [v["name"] for v in cfg["data"]["variables"]]
    lead_times = cfg["data"]["lead_times_hours"]

    frost_cfg = cfg["tasks"]["frost_protection"]
    heat_cfg = cfg["tasks"]["heat_protection"]
    wind_cfg = cfg["tasks"]["wind_power"]

    all_tasks = [
        FrostProtection(
            **{k: v for k, v in frost_cfg.items()
               if k in ("theta_grid", "c_ratio_grid", "scale", "transition_width")}
        ),
        HeatProtection(
            **{k: v for k, v in heat_cfg.items()
               if k in ("theta_grid", "c_ratio_grid", "scale", "transition_width")}
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

    crps_by_model = {"OmniPredictor": {}, "IFS ENS": {}}

    for var in var_names:
        for lt in lead_times:
            pkl_file = results_dir / f"{var}_{lt}h.pkl"
            if not pkl_file.exists():
                logger.warning(f"Missing result: {pkl_file}")
                continue

            with open(pkl_file, "rb") as f:
                result = pickle.load(f)

            p_omni = result["p_omni"]
            p_ifs = result["p_ifs"]
            y_obs = result["y_obs"]
            history = result["history"]

            # CRPS
            crps_omni = mean_crps(p_omni, y_obs, quantile_levels)
            crps_ifs = mean_crps(p_ifs, y_obs, quantile_levels)
            key = f"{var}_{lt}h"
            crps_by_model["OmniPredictor"][lt] = crps_omni
            crps_by_model["IFS ENS"][lt] = crps_ifs
            logger.info(f"{key}: CRPS OmniPred={crps_omni:.4f}, IFS={crps_ifs:.4f}")

            # PIT
            pit_vals, pit_counts = pit_histogram(p_omni, y_obs, quantile_levels)
            logger.info(f"{key}: PIT histogram counts={pit_counts}")

            # Convergence plot
            fig = plot_convergence(history, save_path=str(fig_dir / f"convergence_{key}.png"))
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)

            # Decision cost comparison
            var_tasks = [
                t for t in all_tasks
                if ("wind" in var.lower()) == isinstance(t, WindPowerDispatch)
            ]
            models = {"OmniPredictor": p_omni, "IFS ENS": p_ifs}
            comparison = compare_models(models, y_obs, var_tasks, quantile_levels)
            rankings = rank_models(comparison)
            stability = cross_task_ranking_stability(rankings)
            logger.info(f"{key} rankings: {rankings}")

            fig = plot_ranking_heatmap(
                stability, save_path=str(fig_dir / f"ranking_{key}.png")
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Cross-lead CRPS comparison
    fig = plot_crps_comparison(
        crps_by_model, lead_times, save_path=str(fig_dir / "crps_by_lead.png")
    )
    if fig:
        import matplotlib.pyplot as plt
        plt.close(fig)

    logger.info(f"All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
