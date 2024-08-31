#!bin/bash

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation Fsilu --pbar
CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation Fgelu --pbar
CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation Felu --pbar

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 -cl --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar
CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 -gr --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar

CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch vonenet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar
CUDA_VISIBLE_DEVICES=3 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch debiasedresnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar