#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu_silu --pbar

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu_elu --pbar

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu_gelu --pbar


CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation Frelu_silu --pbar

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation Frelu_elu --pbar
