CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation Frelu_gelu --pbar

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 3
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_elu --device cuda --csv_results_dir rs -iter 5

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_gelu --device cuda --csv_results_dir rs -iter 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_gelu --device cuda --csv_results_dir rs -iter 3
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Frelu_gelu --device cuda --csv_results_dir rs -iter 5

