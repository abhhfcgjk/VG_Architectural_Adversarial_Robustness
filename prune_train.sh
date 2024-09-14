#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 3 --images_count_prune 100 --prune_iters 1 --pbar
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 5 --images_count_prune 100 --prune_iters 1 --pbar
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 10 --images_count_prune 100 --prune_iters 1 --pbar
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 20 --images_count_prune 100 --prune_iters 1 --pbar
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 30 --images_count_prune 100 --prune_iters 1 --pbar
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 2 --images_count_prune 100 --prune_iters 1 --pbar
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 240 --height_prune 180 --kernel_prune 40 --images_count_prune 100 --prune_iters 1 --pbar --debug

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 332 --height_prune 249 --kernel_prune 5 --images_count_prune 100 --prune_iters 1 --pbar 
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 332 --height_prune 249 --kernel_prune 3 --images_count_prune 100 --prune_iters 1 --pbar 
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 120 --height_prune 90 --kernel_prune 1 --images_count_prune 50 --prune_iters 1 --pbar 
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1

CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 120 --height_prune 90 --kernel_prune 1 --images_count_prune 100 --prune_iters 1 --pbar 
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls --prune_iters 1
