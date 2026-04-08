"""Download geoarches model weights from HuggingFace into modelstore/."""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "gcouairon/ArchesWeather"
MODELS = [
    "archesweather-m-seed0",
    "archesweather-m-seed1",
    "archesweather-m-skip-seed0",
    "archesweather-m-skip-seed1",
    "archesweathergen",
]

modelstore = Path("modelstore")

for model in MODELS:
    dest = modelstore / model
    ckpt_dir = dest / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model}...")
    hf_hub_download(REPO_ID, f"{model}_config.yaml", local_dir=dest)
    (dest / f"{model}_config.yaml").rename(dest / "config.yaml")

    hf_hub_download(REPO_ID, f"{model}_checkpoint.ckpt", local_dir=ckpt_dir)
    print(f"  {model} done.")

print("All models downloaded.")
