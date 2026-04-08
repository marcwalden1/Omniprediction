#!/bin/bash
#SBATCH --job-name=omni_arches
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/home06/mwalden/omni_weather/logs/arches_%j.out
#SBATCH --error=/n/home06/mwalden/omni_weather/logs/arches_%j.err

module load Mambaforge/23.11.0-fasrc01
source activate omni_weather

cd /n/home06/mwalden/omni_weather
mkdir -p logs

export OMNI_CONFIG=${OMNI_CONFIG:-config/smoke.yaml}
export OMNI_ARCHES_BATCH_SIZE=4

echo "Using config: $OMNI_CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/run_arches.py
