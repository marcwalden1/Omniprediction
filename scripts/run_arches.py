"""One-time ArchesWeatherGen inference — GPU recommended."""
import logging
import sys
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.arches_runner import run_arches_inference
from data.loader import load_ifs_ens


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["data"]["local_cache_dir"])
    var_names = [v["name"] for v in cfg["data"]["variables"]]

    logger.info("Loading IFS ENS for Arches conditioning...")
    ifs_ds = load_ifs_ens(
        gcs_path=cfg["data"]["ifs_ens_path"],
        variables=var_names,
        year=cfg["data"]["year"],
        region=cfg["data"]["region"],
        lead_times_hours=cfg["data"]["lead_times_hours"],
        local_cache=cache_dir,
    )

    output_zarr = cfg["arches"]["local_zarr"]
    logger.info(f"Running Arches inference → {output_zarr}")
    run_arches_inference(
        ifs_ds=ifs_ds,
        variables=var_names,
        lead_times_hours=cfg["data"]["lead_times_hours"],
        n_samples=cfg["arches"]["n_samples"],
        output_zarr=output_zarr,
        model_name=cfg["arches"]["model_name"],
    )
    logger.info("Arches inference done.")


if __name__ == "__main__":
    main()
