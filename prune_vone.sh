#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune l2 --prune_iters 1 --pbar 
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune l2 --prune_iters 1

# CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune l2 --prune_iters 2 --pbar 
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune l2 --prune_iters 2

# CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.2 -t_prune l2 --prune_iters 1 --pbar 
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.2 -t_prune l2 --prune_iters 1

# CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.05 -t_prune l2 --prune_iters 1 --pbar 
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.05 -t_prune l2 --prune_iters 1

# CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch vonenet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.05 -t_prune l2 --prune_iters 2 --pbar 
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.05 -t_prune l2 --prune_iters 2


CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.05 -t_prune l2 --prune_iters 1 --pbar
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.05 -t_prune l2 --prune_iters 1 --prune_epochs 5

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.05 -t_prune l2 --prune_iters 2 --pbar
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.05 -t_prune l2 --prune_iters 2 --prune_epochs 5

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.05 -t_prune l1 --prune_iters 1 --pbar
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.05 -t_prune l2 --prune_iters 1 --prune_epochs 5

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.05 -t_prune l1 --prune_iters 2 --pbar
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ --resize -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.05 -t_prune l2 --prune_iters 2 --prune_epochs 5
