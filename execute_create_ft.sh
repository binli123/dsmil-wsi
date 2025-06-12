#!/bin/bash
#SBATCH -A hai
#SBATCH --mincpus=18
#SBATCH --gres=gpu:2
#SBATCH -p ihub
#SBATCH --nodelist=gnode103
#SBATCH --mem-per-cpu=3G
#SBATCH --time=90:00:00
#SBATCH --output=/home/karan.padariya/results/create_ft.txt

# Load Miniconda and activate the conda environment
source /home/karan.padariya/miniconda3/etc/profile.d/conda.sh
conda activate dinov2

# Confirm the environment activation
echo "Conda environment activated: $(conda info --envs | grep '*')"

# Copy data from the server
scp -r karan.padariya@gnode096:/ssd_scratch/karan.p /ssd_scratch/

# Run the Python script
python compute_feats.py
# python train_tcga.py

