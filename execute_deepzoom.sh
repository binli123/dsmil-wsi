#!/bin/bash
#SBATCH -A hai
#SBATCH --mincpus=18
#SBATCH --gres=gpu:2
#SBATCH -p ihub
#SBATCH --nodelist gnode096
#SBATCH --mem-per-cpu=3G
#SBATCH --time=90:00:00
#SBATCH --output=/home/karan.padariya/results/deep_zoom_0-17.txt

source /home/karan.padariya/miniconda3/etc/profile.d/conda.h
conda activate dinov2

echo "Conda environment activated: $(conda info --envs | grep '*')"

#scp -r karan.padariya@gnode099:/ssd_scratch/karan.p/HDD1 /ssd_scratch/karan.p
total_files=1053
num_processes=30
chunk_size=$(( (total_files + num_processes - 1) / num_processes )) # Ceiling division

for i in $(seq 0 $((num_processes - 1))); do
    start=$((i * chunk_size))
    end=$((start + chunk_size))
    
    # Make sure 'end' doesn't exceed 'total_files'
    if [ $end -gt $total_files ]; then
        end=$total_files
    fi

    temp=$i
    echo "Process $i: files $start to $end"
    
    python deepzoom_tiler.py --dataset "lung_tcga_tumor" --temp $temp --start $start --end $end --magnifications 2 3 &
done

wait


# python compute_feats.py --dataset "Lungs_10_20_full" 

# python train_tcga.py --dataset "Oral_10_20" 

#python deepzoom_tiler.py --dataset "Oral_5_20" --temp 11 --start 350 --end -1 --magnifications 1 3
#python deepzoom_tiler.py --dataset "Oral_5_20" --temp 4 --start 0 --end 50

