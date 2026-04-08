#!/bin/bash
#SBATCH --job-name=omni_dl_weights
#SBATCH --partition=serial_requeue
#SBATCH --account=kdbrantley_lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/n/home06/mwalden/omni_weather/logs/download_%j.out
#SBATCH --error=/n/home06/mwalden/omni_weather/logs/download_%j.err

module load Mambaforge/23.11.0-fasrc01
source activate omni_weather
cd /n/home06/mwalden/omni_weather

python scripts/download_modelstore.py
