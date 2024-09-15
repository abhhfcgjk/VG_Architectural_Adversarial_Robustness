#!/bin/bash


# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu -gr --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_silu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_silu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_silu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_silu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_elu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_elu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_elu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_elu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_gelu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_gelu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_gelu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation relu_gelu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_silu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_silu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_silu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_silu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 8


# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch wideresnet50 --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch wideresnet50 --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch wideresnet50 --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch wideresnet50 --activation relu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune l1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 3 -prune 0.1 -t_prune l1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune l1
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 8 -prune 0.1 -t_prune l1

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 1 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 3 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 5 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 8 
