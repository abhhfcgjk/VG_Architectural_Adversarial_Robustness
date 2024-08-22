#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fsilu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fsilu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fsilu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fsilu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fgelu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fgelu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fgelu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Fgelu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Felu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Felu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Felu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 --activation Felu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -gr --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -gr --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -gr --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -gr --activation relu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet101 -cl --crop --activation relu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch vonenet101 --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch vonenet101 --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch vonenet101 --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch vonenet101 --activation relu --device cuda --csv_results_dir rs -iter 8

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch debiasedresnet101 --activation relu --device cuda --csv_results_dir rs -iter 1
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch debiasedresnet101 --activation relu --device cuda --csv_results_dir rs -iter 3
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch debiasedresnet101 --activation relu --device cuda --csv_results_dir rs -iter 5
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch debiasedresnet101 --activation relu --device cuda --csv_results_dir rs -iter 8

CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 3
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 8
