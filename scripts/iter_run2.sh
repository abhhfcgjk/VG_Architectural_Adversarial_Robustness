#!/bin/bash

for i in {0..6}; do
sbatch -W scripts/run_dali2.sh fc
wait
sbatch -W scripts/run_dali2.sh
wait
done