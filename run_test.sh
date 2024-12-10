#!/bin/bash

#SBATCH --ntasks-per-gpu=6
#SBATCH --job-name=28i_mel-dbcnn
#SBATCH --time=1-1:00
#SBATCH --dependency=singleton
#SBATCH --gpus=8

set -e

mapfile -t args < /home/maindev/28i_mel/Linearity/options/args.txt

metric_name="KonCept"
dir_name="."
config_dir="/home/maindev/28i_mel/Linearity/presets"

declare -a iterations=(1 3 5 8)

echo "Metric ${metric_name}"
for arg in "${args[@]}"; do
    config_name=${arg//|-/}
    arg=${arg//|/ }
    config_name="$(python3 build_config.py --config_dir $config_dir $arg)"
    printf "Config: ${config_name}\n"
    mv $config_name $config_dir
    printf "${metric_name} ${arg}.\n Starting job\n"

    CUDA_VISIBLE_DEVICES=3 python main.py --config presets/"${config_name}" >> "${metric_name}-${config_name//.yaml/}".out &
    wait
    # srun --exclusive --ntasks 4 -G 1 \
    # --container-image /scratch/amoskalenko/users/28i_mel/images/28i_mel+dbcnn+arch_rob.sqsh \
    # --container-mounts /scratch/amoskalenko/users/28i_mel/vanDBCNN:/workdir/vg bash \
    # -c "cd vg && \
    # pip install -r requirements.txt && \
    # python main.py --config presets/'${config_name}' > DBCNN'${config_name//.yaml/}'.out" &
    
done
