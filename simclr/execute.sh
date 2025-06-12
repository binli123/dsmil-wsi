#!/bin/bash
#SBATCH -A hai
#SBATCH --mincpus=36
#SBATCH --gres=gpu:4
#SBATCH -p ihub
#SBATCH --nodelist=gnode100
#SBATCH --mem-per-cpu=3G
#SBATCH --time=90:00:00
#SBATCH --output=/home/karan.padariya/results/train_rensnet_ORCHID.txt

# Load Miniconda and activate the conda environment
source /home/karan.padariya/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest

# Confirm the environment activation
echo "Conda environment activated: $(conda info --envs | grep '*')"


# Run the Python script
python run.py --dataset 'Oral_10_20' --multiscale 0

