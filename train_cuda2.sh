#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar
CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation silu --pbar
CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation elu --pbar
CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation gelu --pbar

CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar
CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune l1 --pbar
CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune l2 --pbar
