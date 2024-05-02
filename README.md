## System
```
            .-/+oossssoo+/-.               
         :+ssssssssssssssssss+:            ------------ 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 18.04.6 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Kernel: 5.4.0-84-generic 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Uptime: 17 hours, 38 mins 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Packages: 1786 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Shell: bash 4.4.20 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Resolution: 1920x1080 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   DE: GNOME 3.28.4 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   WM: GNOME Shell 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   WM Theme: Adwaita 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Theme: Ambiance [GTK2/3] 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Icons: Ubuntu-mono-dark [GTK2/3] 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    Terminal: gnome-terminal 
  +sssssssssdmydMMMMMMMMddddyssssssss+     CPU: AMD Ryzen 5 1600 (12) @ 3.200GHz 
   /ssssssssssshdmNNNNmyNMMMMhssssss/      GPU: NVIDIA GeForce GTX 1060 6GB 
    .ossssssssssssssssssdMMMNysssso.       Memory: 2771MiB / 16010MiB 
      -+sssssssssssssssssyyyssss+- 
         :+ssssssssssssssssss+:                                   
            .-/+oossssoo+/-. 
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

conda 23.7.4
python 3.7.16
pip 22.3.1

## Apex installation
```
git clone https://github.com/NVIDIA/apex
```
```
cd apex
```
```
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Code a few lines in package and run
```
python setup.py install --cpp_ext --cuda_ext
```
Rename apex folder in /anaconda3/envs/smooth3.7/lib/python3.7/site-packages to apex-0.1
```
cd ..
rm -rf apex
```

## Training on KonIQ-10k
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch resnet34 --loss_type norm-in-norm --p 1 --q 2 --activation silu --pbar
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --feature_model debiased --mgamma 0.1 --debug
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 4 -e 5 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar
```
## Testing
```
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet34 --activation silu --device cpu --csv_results_dir rs -iter 10
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1 --feature_model debiased -mg 0.1
```

## Visualization
```
tensorboard --logdir=runs --port=6006
```
## Results

| arch | activation | eps 2 | eps 4 | eps 6 | eps 8 | eps 10 |
|------|:----------:|:-----:|:-----:|:-----:|:-----:|-------:|
|resnet34|relu|783.23204070329|1563.6732801795|2341.83929860591|3117.28492379188|3890.36685228347|
|resnet34|silu|783.28941017389|1563.97335231304|2342.45024621486|3118.20954084396|3891.54888689518|
|wideresnet50|relu|783.2502014935017|1563.7392178177834|2341.9534787535667|3117.4952164292336|3890.666365623474|
|resnet50|relu|783.2514122128487|1563.7654811143875|2341.969683766365|3117.523342370987|3890.707716345787|
|resnet34|relu|783.2173258066177|1563.6410564184189|2341.7897522449493|3117.240220308304|3890.296071767807|

| arch | activation | degree | eps 2 | eps 4 | eps 6 | eps 8 | eps 10 | SROCC |
|------|:----------:|:------:|:-----:|:-----:|:-----:|:-----:|:------:|------:|
|	resnet34|	silu|	10^4|	783.28|	1563.96|	2342.46|	3118.32|	3891.82|	0.91|
|	resnet34	|relu	|10^4	|783.22	|1563.64	|2341.79	|3117.24	|3890.3	|0.92|
|	resnet50	|relu	|10^4	|783.25	|1563.77	|2341.97	|3117.52	|3890.71	|0.94|
|	wideresnet50|	relu|	10^4|	783.25|	1563.74|	2341.96	|3117.5	|3890.67|	0.94|
									



## Links
[KonIQ-10k](https://drive.google.com/file/d/13KlUl_Uo68MDjL_ef7INQHf_waDZf4R9/view?usp=drive_link)<br>
[Checkpoints-SiLU](https://drive.google.com/file/d/19sbNdE7EJDQScCWgPuQpxgptYt1YckM0/view?usp=drive_link)<br>
[Checkpoints-ReLU](https://drive.google.com/file/d/1pte9VqUfsD3Eu0DSNQYKQpbtJep2zxEB/view?usp=drive_link)<br>
[Checkpoints-SiLU_res34](https://drive.google.com/file/d/1OrU0zi8-TWI_MkE_1OetBKG-EvQ8hGpX/view?usp=drive_link)<br>
[Checkpoints-ReLU_res34](https://drive.google.com/file/d/1kDGQ96qYbZuXXpqiT-BD-pk8TY_WNtkS/view?usp=drive_link)<br>
[Checkpoints-ReLU-SiLU_res34](https://drive.google.com/file/d/1F-9J2R9j6ID5ln-ZPPcDPzUlXZX0PuFr/view?usp=drive_link)<br>
