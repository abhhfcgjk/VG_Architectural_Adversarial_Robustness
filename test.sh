#!/bin/bash

python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation Fsilu --device cuda --csv_results_dir rs -iter 1
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation silu --device cuda --csv_results_dir rs -iter 1
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu_silu --device cuda --csv_results_dir rs -iter 1

python main.py --dataset_path ./NIPS_test/ --resize -arch resnet18 --activation relu --device cuda --csv_results_dir rs -iter 1
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet34 --activation relu --device cuda --csv_results_dir rs -iter 1
python main.py --dataset_path ./NIPS_test/ --resize -arch wideresnet50 --activation relu --device cuda --csv_results_dir rs -iter 1

python main.py --dataset_path ./NIPS_test/ --resize -arch vonenet50 --activation relu --device cuda --csv_results_dir rs -iter 1
python main.py --dataset_path ./NIPS_test/ --resize -arch advresnet50 --activation relu --device cuda --csv_results_dir rs -iter 1
