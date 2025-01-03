#!/bin/bash

for i in {0..6}; do
sbatch -W scripts/run_dali.sh fc
wait
sbatch -W scripts/run_dali.sh
wait
done