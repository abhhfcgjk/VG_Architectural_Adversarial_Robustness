#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation elu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation elu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation elu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation elu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation gelu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation gelu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation gelu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation gelu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation sile --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation silu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation silu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation silu --device cuda --csv_results_dir rs -iter 8


# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 3 -prune 0.1 -t_prune pls
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune pls
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 8 -prune 0.1 -t_prune pls

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune l1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 3 -prune 0.1 -t_prune l1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune l1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 8 -prune 0.1 -t_prune l1

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune l2
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 3 -prune 0.1 -t_prune l2
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune l2
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 8 -prune 0.1 -t_prune l2

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -cl --device cuda --csv_results_dir rs -iter 1 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -cl --device cuda --csv_results_dir rs -iter 3 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -cl --device cuda --csv_results_dir rs -iter 5 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -cl --device cuda --csv_results_dir rs -iter 8 

