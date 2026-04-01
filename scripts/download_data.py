"""Download IFS ENS and ERA5 data from GCS to local zarr cache."""
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_era5, load_ifs_ens


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["data"]["local_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    region = cfg["data"]["region"]
    year = cfg["data"]["year"]
    var_names = [v["name"] for v in cfg["data"]["variables"]]
    lead_times = cfg["data"]["lead_times_hours"]

    logger.info("Downloading ERA5...")
    era5 = load_era5(
        gcs_path=cfg["data"]["era5_path"],
        variables=var_names,
        year=year,
        region=region,
        local_cache=cache_dir,
    )
    logger.info(f"ERA5: {era5}")

    logger.info("Downloading IFS ENS...")
    ifs = load_ifs_ens(
        gcs_path=cfg["data"]["ifs_ens_path"],
        variables=var_names,
        year=year,
        region=region,
        lead_times_hours=lead_times,
        local_cache=cache_dir,
    )
    logger.info(f"IFS ENS: {ifs}")
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
