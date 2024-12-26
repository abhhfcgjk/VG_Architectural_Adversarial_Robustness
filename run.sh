#!/bin/bash

#SBATCH --ntasks-per-gpu=6
#SBATCH --job-name=28i_mel-dbcnn
#SBATCH --time=1-1:00
#SBATCH --gpus=8
srun --exclusive --ntasks 1 -G 1 \
--container-image /scratch/amoskalenko/users/28i_mel/images/28i_mel+dbcnn+arch_rob.sqsh \
--container-mounts /scratch/amoskalenko/users/28i_mel/Linearity:/workdir/vg bash \
-c "cd vg && \
pip install -r requirements.txt && \
python main.py --config presets/'${config_name}' > Linearity'${config_name//.yaml/}'.out" &
