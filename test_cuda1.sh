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

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 1 
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 3 
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 5 
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cp --activation relu --device cuda --csv_results_dir rs -iter 8 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 120 --height_prune 90 --kernel_prune 1 --images_count_prune 50 --prune_iters 1 --pbar
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

CUDA_VISIBLE_DEVICES=1 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 -cl --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 120 --height_prune 90 --kernel_prune 1 --images_count_prune 50 --prune_iters 1 --pbar
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 -cl --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1
