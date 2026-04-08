#!/bin/bash
#SBATCH --job-name=omni_train
#SBATCH --partition=serial_requeue
#SBATCH --account=kdbrantley_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/home06/mwalden/omni_weather/logs/train_%j.out
#SBATCH --error=/n/home06/mwalden/omni_weather/logs/train_%j.err

module load Mambaforge/23.11.0-fasrc01
source activate omni_weather
cd /n/home06/mwalden/omni_weather
mkdir -p logs

export OMNI_CONFIG=${OMNI_CONFIG:-config/smoke.yaml}
echo "Using config: $OMNI_CONFIG"
python scripts/train_omni.py
