#!/bin/bash

python main.py --dataset_path ./NIPS_test/  -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/  -arch resnet50 --activation Fsilu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/  -arch resnet50 --activation silu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu_silu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_silu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation gelu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation elu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Fgelu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Felu --device cuda --csv_results_dir rs -iter 5

python main.py --dataset_path ./NIPS_test/ -arch resnet18 --activation relu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/ -arch resnet34 --activation relu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/  -arch wideresnet50 --activation relu --device cuda --csv_results_dir rs -iter 5

python main.py --dataset_path ./NIPS_test/  -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 5
python main.py --dataset_path ./NIPS_test/  -arch advresnet50 --activation relu --device cuda --csv_results_dir rs -iter 5

python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune pls
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune l1
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 5 -prune 0.1 -t_prune l2
