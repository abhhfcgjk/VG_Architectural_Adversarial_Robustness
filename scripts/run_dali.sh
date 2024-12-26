#!/bin/bash

#SBATCH --ntasks-per-gpu=6
#SBATCH --job-name=28i_mel-dbcnn
#SBATCH --time=1-1:00
#SBATCH --gpus=8
set -e

mapfile -t args < /scratch/amoskalenko/users/28i_mel/vanKonCept/options/train.txt

metric_name="KonCept"
dir_name="."
config_dir="/scratch/amoskalenko/users/28i_mel/vanKonCept/presets"

echo "Metric ${metric_name}"
for arg in "${args[@]}"; do
    config_name=${arg//|-/}
    arg=${arg//|/ }
    config_name="$(python3 /scratch/amoskalenko/users/28i_mel/vanKonCept/scripts/build_config.py --config_dir $config_dir $arg)"
    printf "Config: ${config_name}"
    mv $config_name $config_dir
    printf "KonCept ${arg}.\n Starting job"
    
    srun --exclusive --ntasks 1 -G 1 \
    --container-image /scratch/amoskalenko/users/28i_mel/images/28i_mel+dbcnn+arch_rob.sqsh \
    --container-mounts /scratch/amoskalenko/users/28i_mel/vanKonCept:/workdir/vg bash \
    -c "cd vg && \
    pip install -r requirements.txt && \
    python main.py --config presets/'${config_name}' > KonCept'${config_name//.yaml/}'.out" &
done
wait