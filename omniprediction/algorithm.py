"""OmniPrediction main loop: W-calibration + C-multiaccuracy (exceedance-probability rep)."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _random_init(N: int, d: int) -> np.ndarray:
    """Generate (N, d) random exceedance probabilities sorted decreasingly per row."""
    rng = np.random.default_rng()
    p = rng.uniform(0.0, 1.0, size=(N, d))
    p = np.sort(p, axis=1)[:, ::-1]  # sort decreasingly: p_0 >= p_1 >= ... >= p_{d-1}
    return p.astype(np.float64)


def run_omniprediction(
    p_ifs: np.ndarray,       # (N, d) — IFS exceedance probs in [0, 1]
    p_arches: np.ndarray,    # (N, d) — Arches exceedance probs in [0, 1]
    y_obs_norm: np.ndarray,  # (N,) — normalized observations
    losses: list,            # flat list of DecisionTask instances (one per param combo)
    lam: float,
    d: int,
    epsilon: float = 0.01,
    eta: float = 0.01,
    max_iterations: int = 1000,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Run OmniPrediction algorithm (exceedance-probability representation).

    Algorithm:
      p_0 = random exceedance probs in [0, 1]^d, sorted decreasingly

      for t in range(max_iterations):
          # W-calibration
          for each loss:
              k_t = loss.k_ell(p_t)
              w   = loss.delta_L(k_t)
              if E[<w, s(y)>] > ε:
                  p_t = clip(p_t + η*w, 0, 1); break

          # C-multiaccuracy (no k_ell — use hypothesis predictions directly)
          for h in {p_ifs, p_arches}:
              if E[<p_h, s(y) - p_t>] > ε:
                  p_t = clip(p_t + η*p_h, 0, 1); break

          if no update: converged

    Returns:
        p_final: (N, d) converged exceedance-probability predictions
        history: dict with iteration logs
    """
    tau = (np.arange(d) - d // 2) * lam  # (d,) normalized thresholds
    N = y_obs_norm.shape[0]

    # Observation summary: s_k(y) = 1[y >= tau_k]
    s_y = (y_obs_norm[:, np.newaxis] >= tau).astype(np.float64)  # (N, d)

    p_t = _random_init(N, d)
    history = {"violations": [], "n_updates": 0, "converged": False}
    hypotheses = {"ifs": p_ifs, "arches": p_arches}

    for t in range(max_iterations):
        updated = False

        # W-calibration: compose k_ell with delta_L
        for loss in losses:
            k_t = loss.k_ell(p_t)        # (N,)
            w = loss.delta_L(k_t)        # (N, d)
            violation = float(np.mean(np.sum(w * s_y, axis=1)))

            history["violations"].append({
                "t": t,
                "type": "W",
                "task": type(loss).__name__,
                "violation": violation,
            })

            if verbose and t % 100 == 0:
                logger.debug("t=%d W-cal %s violation=%.4f", t, type(loss).__name__, violation)

            if violation > epsilon:
                p_t = np.clip(p_t + eta * w, 0.0, 1.0)
                history["n_updates"] += 1
                updated = True
                break

        if updated:
            if verbose and t % 100 == 0:
                logger.info("t=%d update (W-cal), total=%d", t, history["n_updates"])
            continue

        # C-multiaccuracy: hypothesis predictions used directly as weight (no k_ell)
        for h_name, p_h in hypotheses.items():
            violation = float(np.mean(np.sum(p_h * (s_y - p_t), axis=1)))

            history["violations"].append({
                "t": t,
                "type": f"C_{h_name}",
                "violation": violation,
            })

            if violation > epsilon:
                p_t = np.clip(p_t + eta * p_h, 0.0, 1.0)
                history["n_updates"] += 1
                updated = True
                break

        if not updated:
            history["converged"] = True
            if verbose:
                logger.info(
                    "Converged after %d iterations, %d updates.", t + 1, history["n_updates"]
                )
            break

        if verbose and t % 100 == 0:
            logger.info("t=%d update (C-multiacc), total=%d", t, history["n_updates"])

    return p_t, history
