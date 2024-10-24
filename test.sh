#!/bin/bash

CUDA_VISIBLE_DEVICES=3 nohup uv run main.py -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 1 
CUDA_VISIBLE_DEVICES=3 nohup uv run main.py -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 3 
CUDA_VISIBLE_DEVICES=3 nohup uv run main.py -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 5 
CUDA_VISIBLE_DEVICES=3 nohup uv run main.py -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 8 
