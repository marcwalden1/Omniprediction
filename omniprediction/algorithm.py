"""
OmniPrediction main loop: W-calibration + C-multiaccuracy.

Algorithm (from plan):
  p_0 = IFS ENS empirical quantiles

  for t in range(max_iterations):
      for each (task, θ, c_ratio) in loss param grid:
          # W-calibration check
          k_t = argmin_a E_{y~CDF(p_t)}[cost(a, y)]
          w = ΔL(y_obs, θ, c_ratio; action=k_t) * 1_d
          if E[⟨w, y*1_d - p_t⟩] > ε:
              p_{t+1} = clip(p_t + η*w, -1, 1) then isotonic
              break

          # C-multiaccuracy check
          for h in {IFS, Arches}:
              k_h = argmin_a E_{y~h_ensemble}[cost(a, y)]
              c_h = ΔL(y_obs, θ, c_ratio; action=k_h) * 1_d
              if E[⟨c_h, y*1_d - p_t⟩] > ε:
                  p_{t+1} = clip(p_t + η*c_h, -1, 1) then isotonic
                  break
      else:
          break  # converged
"""
import logging
from typing import Optional
import numpy as np

from omniprediction.predictor import OmniPredictor
from omniprediction.action_solver import optimal_actions
from omniprediction.weights import calibration_weight, multiaccuracy_weight

logger = logging.getLogger(__name__)


def run_omniprediction(
    p_ifs: np.ndarray,           # (N, d) — IFS ENS quantiles, normalized
    p_arches: np.ndarray,        # (N, d) — Arches quantiles, normalized
    y_obs_norm: np.ndarray,      # (N,) — normalized observations
    tasks: list,                 # list of DecisionTask instances
    quantile_levels: list[float],
    epsilon: float = 0.01,
    eta: float = 0.01,
    max_iterations: int = 1000,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Run OmniPrediction algorithm.

    Returns:
        p_final: (N, d) converged quantile predictions
        history: dict with iteration logs
    """
    quantile_levels_arr = np.array(quantile_levels)
    d = len(quantile_levels)
    N = y_obs_norm.shape[0]

    predictor = OmniPredictor(p_ifs.copy(), quantile_levels)
    history = {"violations": [], "n_updates": 0, "converged": False}

    # Pre-compute hypothesis quantiles
    hypotheses = {
        "ifs": p_ifs,
        "arches": p_arches,
    }

    for t in range(max_iterations):
        updated = False

        for task in tasks:
            # Build cost function wrapper that matches task.cost signature:
            # cost_fn(action, y_1d, params) → (n_quad,)
            def make_cost_fn(task_):
                def cost_fn(action, y_1d, params):
                    return task_.cost(action, y_1d, params)
                return cost_fn

            cost_fn = make_cost_fn(task)

            for params in task.param_grid:
                p_curr = predictor.get_quantiles()  # (N, d)

                # W-calibration check
                k_t = optimal_actions(
                    p_curr, quantile_levels_arr, cost_fn, params, task.n_actions
                )  # (N,)

                # Compute ΔL for each sample using its optimal action
                delta_l = _compute_delta_L_per_sample(task, y_obs_norm, k_t, params, N)

                w, violation = calibration_weight(y_obs_norm, p_curr, delta_l, d)

                if verbose and t % 50 == 0:
                    logger.debug(
                        f"t={t}, task={type(task).__name__}, params={params}, "
                        f"W-violation={violation:.4f}"
                    )

                history["violations"].append(
                    {
                        "t": t,
                        "type": "W",
                        "task": type(task).__name__,
                        "params": str(params),
                        "violation": violation,
                    }
                )

                if violation > epsilon:
                    predictor.update(w, eta)
                    history["n_updates"] += 1
                    updated = True
                    break

                # C-multiaccuracy check for each hypothesis
                for h_name, p_h in hypotheses.items():
                    k_h = optimal_actions(
                        p_h, quantile_levels_arr, cost_fn, params, task.n_actions
                    )  # (N,)

                    delta_l_h = _compute_delta_L_per_sample(task, y_obs_norm, k_h, params, N)
                    c_h, violation_h = multiaccuracy_weight(
                        y_obs_norm, p_curr, delta_l_h, d
                    )

                    history["violations"].append(
                        {
                            "t": t,
                            "type": f"C_{h_name}",
                            "task": type(task).__name__,
                            "params": str(params),
                            "violation": violation_h,
                        }
                    )

                    if violation_h > epsilon:
                        predictor.update(c_h, eta)
                        history["n_updates"] += 1
                        updated = True
                        break

                if updated:
                    break

            if updated:
                break

        if not updated:
            history["converged"] = True
            if verbose:
                logger.info(
                    f"Converged after {t + 1} iterations, {history['n_updates']} updates."
                )
            break

        if verbose and t % 100 == 0:
            logger.info(
                f"Iteration {t}/{max_iterations}, updates so far: {history['n_updates']}"
            )

    return predictor.get_quantiles(), history


def _compute_delta_L_per_sample(
    task,
    y_obs_norm: np.ndarray,
    actions: np.ndarray,  # (N,) integer actions
    params: dict,
    N: int,
) -> np.ndarray:
    """
    Compute ΔL(y_obs, θ, c; action=k) for each sample.

    For samples with the same action, batch-compute delta_L.
    """
    delta_l = np.zeros(N)
    unique_actions = np.unique(actions)
    for a in unique_actions:
        mask = actions == a
        delta_l[mask] = task.delta_L(y_obs_norm[mask], int(a), params)
    return delta_l
