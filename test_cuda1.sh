#!/bin/bash


# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 8

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 3
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 5
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 8
