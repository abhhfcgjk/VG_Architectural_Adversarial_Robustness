#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 8

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet18 --activation relu --device cuda --csv_results_dir rs -iter 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet18 --activation relu --device cuda --csv_results_dir rs -iter 3
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet18 --activation relu --device cuda --csv_results_dir rs -iter 5
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet18 --activation relu --device cuda --csv_results_dir rs -iter 8

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet34 --activation relu --device cuda --csv_results_dir rs -iter 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet34 --activation relu --device cuda --csv_results_dir rs -iter 3
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet34 --activation relu --device cuda --csv_results_dir rs -iter 5
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet34 --activation relu --device cuda --csv_results_dir rs -iter 8

