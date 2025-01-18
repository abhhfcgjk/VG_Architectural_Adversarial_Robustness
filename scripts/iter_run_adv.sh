#!/bin/bash

sbatch -W scripts/run_dali-adv.sh fc
wait
sbatch -W scripts/run_dali-adv.sh
wait
