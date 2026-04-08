#!/bin/bash
set -e

module load Mambaforge/23.11.0-fasrc01

conda create -y -n omni_weather python=3.11

conda run -n omni_weather pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

conda run -n omni_weather pip install tensordict

conda run -n omni_weather pip install \
    "numpy>=1.24" "xarray>=2023.1" "zarr>=2.14,<3" "gcsfs>=2023.1" \
    "scikit-learn>=1.2" "scipy>=1.10" "matplotlib>=3.7" "pyyaml>=6.0" "pytest>=7.0"

conda run -n omni_weather pip install "git+https://github.com/INRIA/geoarches.git"

echo "Done. Activate with: module load Mambaforge/23.11.0-fasrc01 && source activate omni_weather"
