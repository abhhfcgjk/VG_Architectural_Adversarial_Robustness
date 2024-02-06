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
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize -rs_h 256 -rs_w 256 --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch resnext101_32x8d --loss_type norm-in-norm --p 1 --q 2 --activation silu
```
## Visualization
```
tensorboard --logdir=runs --port=6006
```
## Links
[KonIQ-10k](https://drive.google.com/file/d/13KlUl_Uo68MDjL_ef7INQHf_waDZf4R9/view?usp=drive_link)<br>
[Checkpoints-SiLU](https://drive.google.com/file/d/19sbNdE7EJDQScCWgPuQpxgptYt1YckM0/view?usp=drive_link)<br>
[Checkpoints-ReLU](https://drive.google.com/file/d/1pte9VqUfsD3Eu0DSNQYKQpbtJep2zxEB/view?usp=drive_link)
