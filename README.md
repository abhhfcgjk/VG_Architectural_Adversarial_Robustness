## System
## Training on KonIQ-10k
```
CUDA_VISIBLE_DEVICES=2 nohup uv run train.py > koncept_original.out&
```
## Testing
```
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Fsilu --device cuda --csv_results_dir rs -iter 1 --model KonCept
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet34 --activation silu --device cuda --csv_results_dir rs -iter 10
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls
```
## Links
