# OmniPrediction Weather Forecasting

Implementation of the OmniPrediction algorithm with a fixed 2-element hypothesis class: **IFS ENS** and **ArchesWeatherGen**. The algorithm iteratively calibrates ensemble quantile forecasts to be simultaneously good across multiple downstream decision tasks (frost protection, heat protection, wind power dispatch).

## Architecture

```
omniprediction/   Core algorithm: algorithm.py (W-cal + C-multiaccuracy loop),
                  predictor.py (quantile state + isotonic projection),
                  action_solver.py, weights.py
data/             loader.py (ERA5 + IFS ENS from gs://weatherbench2),
                  arches_runner.py (real geoarches diffusion inference),
                  normalization.py
tasks/            DecisionTask base + FrostProtection, HeatProtection, WindPowerDispatch
scripts/          run_arches.py (generate Arches cache, GPU),
                  train_omni.py (run algorithm, CPU),
                  evaluate.py (CRPS, PIT, rankings)
config/           default.yaml, smoke.yaml, arches_quick.yaml
modelstore/       Pre-downloaded geoarches weights (~3 GB, 723M params total) — gitignored
plots/            figures.py
```

## Two-step workflow

The Arches cache must exist before training. `train_omni.py` hard-fails if it can't load the zarr (this is intentional — falling back to IFS-as-Arches breaks the algorithm by making both hypotheses identical).

```bash
# Step 1: generate Arches forecast cache (GPU strongly required)
OMNI_CONFIG=config/<cfg>.yaml python scripts/run_arches.py

# Step 2: train OmniPredictor (CPU-feasible)
OMNI_CONFIG=config/<cfg>.yaml python scripts/train_omni.py

# Step 3: evaluate
OMNI_CONFIG=config/<cfg>.yaml python scripts/evaluate.py
```

Config selection via the `OMNI_CONFIG` env var; defaults to `config/default.yaml`.

## Configs

| Config | Purpose | Region | Time | Leads | Members |
|---|---|---|---|---|---|
| `default.yaml` | Full run | Europe (35–72°N) | All of 2021 | 24/48/72/120/168h | 50 |
| `smoke.yaml` | End-to-end smoke | UK (51–53°N) | 7 days | 24h | 10 |
| `arches_quick.yaml` | Minimal sanity | UK (51–53°N) | 2 days | 24h | 1 |

All configs use **1.5° resolution** to match Arches's native grid.

## Arches inference (run_arches.py)

The diffusion model (`archesweathergen`, 384M params) wraps an averaged ensemble of 4 deterministic models (`archesweather-m-*`, 4× 85M params). All 5 models are loaded simultaneously (~3 GB RAM).

**CPU is not viable** — 25 diffusion denoising steps × 723M params per sample. Smoke test would take days; full run would take months. Always run on GPU. Set `OMNI_ARCHES_BATCH_SIZE=4` (env var) for better GPU utilization than the default of 1.

The cache zarr structure (after the step_timestamp fix) is `(time, prediction_timedelta, latitude, longitude, member)` where `time` is the **init time** (matching IFS convention) and `prediction_timedelta` carries the lead.

`_is_valid_arches_store()` validates the cache before reuse; invalid caches are moved to `*.broken.<timestamp>` rather than deleted.

## Algorithm internals

`run_omniprediction()` in `omniprediction/algorithm.py`:
1. Initializes `p_t = p_ifs` (normalized to [-1, 1])
2. For each iteration, checks two violation types against each (task, params) combo:
   - **W-calibration** — uses optimal action under current `p_t`
   - **C-multiaccuracy** — uses optimal action under each hypothesis (IFS, Arches)
3. If any violation > ε, applies `p_t ← isotonic(clip(p_t + η·w, -1, 1))` and breaks to next iteration
4. Converges when all violations ≤ ε

`OmniPredictor._enforce_monotone()` re-applies isotonic regression after every update to keep quantiles monotonic.

Temperature variables use Frost + Heat tasks. Wind variables use WindPower task. This dispatch is in `train_omni.py` based on variable name.

## Data sources

Both ERA5 and IFS ENS are streamed from `gs://weatherbench2` (anonymous GCS access works) and cached locally as zarr v2 in `cfg.data.local_cache_dir`. Cache filenames encode region/resolution/time-window so config changes automatically invalidate the cache.

`arches_runner._load_era5_init_states()` separately streams the 1440×721 ERA5 from GCS for Arches initialization (without local caching) and regrids to the 1.5° global grid.

## Gotchas

- **CLAUDE.md is auto-loaded** — keep it concise. Don't put one-time investigation notes here.
- **Don't restore the IFS-as-Arches fallback** in `train_omni.py`. Hard-failing is intentional.
- **`step_timestamp` in `arches_runner._run_real_arches`** must be `batch["timestamp"].cpu()` (init time), not init+lead. Using valid time produces sparse NaN-filled zarrs for multi-lead configs because `xr.concat` along `prediction_timedelta` does an outer join on the time dimension.
- **`OMP_NUM_THREADS=1` is set by default** in `arches_runner.py` to avoid threading conflicts. Override via env var if running CPU experiments.
- **Lead times must be multiples of 24h** for real Arches inference (`_run_real_arches` raises). The diffusion model rolls out in 24h steps.
- **Member dim naming**: IFS uses `number`/`realization`/`member`; Arches output uses `member`. `extract_quantiles()` and `_run_fallback_arches()` handle the variants.
