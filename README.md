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
CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch resnet34 --loss_type norm-in-norm --p 1 --q 2 --activation silu --pbar
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --feature_model debiased --mgamma 0.1 --debug
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 4 -e 5 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 -arch inceptionresnet --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept -rs_h 384 -rs_w 512 --debug
CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-5 -bs 4 -e 60 --ft_lr_ratio 0.1 -arch inceptionresnet --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept -rs_h 384 -rs_w 512 --debug

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 70 --ft_lr_ratio 0.1 -arch inceptionresnet --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept -rs_h 384 -rs_w 512

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 4 -e 70 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept
```
## Testing
```
python main.py --dataset_path ./NIPS_test/ -arch resnet50 --activation Fsilu --device cuda --csv_results_dir rs -iter 1 --model KonCept
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet34 --activation silu --device cuda --csv_results_dir rs -iter 10
python main.py --dataset_path ./NIPS_test/ --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls
```

## Visualization
```
tensorboard --logdir=runs --port=6006
```
## Results

<!-- | arch | activation | eps 2 | eps 4 | eps 6 | eps 8 | eps 10 |
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
|	wideresnet50|	relu|	10^4|	783.25|	1563.74|	2341.96	|3117.5	|3890.67|	0.94| -->
									
|arch|activation|attack|iterations|eps 2|eps 4|eps 6|eps 8|eps 10|SROCC|
|----|:--------:|:----:|:--------:|:---:|:---:|:---:|:---:|:----:|:----:|
|resnet50-Linearity|relu|IFGSM|1.0|0.317411637289097|0.271145925804271|0.203409666328665|0.135890861648704|0.0711905868014346|0.907342712845833|
|resnet50-Linearity|Fsilu|IFGSM|1.0|0.163610402542496|0.229744249788945|0.260979490454468|0.272827489772825|0.273075761617867|0.805964034119172|
|resnet50-Linearity|silu|IFGSM|1.0|0.259757468717855|0.23818706373003|0.190764396984096|0.139550987124785|0.0886347675207367|0.897967910330036|
|wideresnet50-Linearity|relu|IFGSM|1.0|0.440476591477283|0.384431273431617|0.301024952409307|0.224072102553711|0.155612114483452|0.914137004748674|
|vonenet50-Linearity|relu|IFGSM|1.0|0.145615089197789|0.205410044566726|0.232540992669077|0.242709561170841|0.24332868292649|0.858145411487797|
|advresnet50-Linearity|relu|IFGSM|1.0|0.0611025438093723|0.0946663652016859|0.11539471999587|0.129646397657898|0.139469237948142|0.854452963192787|
|resnet50-Linearity|relu_silu|IFGSM|1.0|0.230368264512998|0.216448610724966|0.184115635267945|0.148635853859261|0.111172278951117|0.907032739330434|
|resnet18-Linearity|relu|IFGSM|1.0|0.298096763164257|0.293965902695312|0.26022545658979|0.22359577736597|0.189172568807873|0.895401377115908|
|resnet34-Linearity|relu|IFGSM|1.0|0.218251315115138|0.21347787263771|0.188101582467376|0.156821526633806|0.123871338665881|0.90180907612445|
|resnet50-Linearity+prune=0.1pls|relu|IFGSM|1.0|0.202443460903232|0.167075693101577|0.118835222943282|0.073644110824028|0.0323004043994797|0.90534272521054|
|resnet50-Linearity+prune=0.1l1|relu|IFGSM|1.0|0.208643864374943|0.178530992987175|0.134158508733606|0.0910499020984101|0.0505680989874457|0.906934696593827|
|resnet50-Linearity+prune=0.1l2|relu|IFGSM|1.0|0.238957705320767|0.188690965305981|0.127481952603284|0.0701611496486072|0.0171878938957012|0.907126080398333|
|resnet50-Linearity|gelu|IFGSM|1.0|0.403301606201018|0.393757472064734|0.344268353823393|0.288486292875002|0.231876088272428|0.905414880534313|
|resnet50-Linearity|elu|IFGSM|1.0|0.22330776541684|0.201555592621671|0.16927916985953|0.135975865788519|0.10115349592574|0.908696613023034|
|resnet50-Linearity|Fgelu|IFGSM|1.0|0.573644929308788|0.790752127526658|0.875399428139593|0.901769387388634|0.891817010123757|0.830302889183858|
|resnet50-Linearity|Felu|IFGSM|1.0|0.109893066494807|0.112317684632482|0.109976290284704|0.108697793749888|0.107717667291837|0.836819747430995|
|debiasedresnet50-Linearity|relu|IFGSM|1.0|0.115846801627178|0.104204860041715|0.0897594529607334|0.0766619504281097|0.0648316974973126|0.900750465090084|
|shaperesnet50-Linearity|relu|IFGSM|1.0|0.179624649010126|0.171043458136231|0.149689439459143|0.127254819449897|0.105996555911404|0.900756089344851|
|textureresnet50-Linearity|relu|IFGSM|1.0|0.267632329718235|0.243983856159449|0.201300531996576|0.158704673919841|0.118445381456513|0.90411785434087|
|inceptionresnet-KonCept|Fsilu|IFGSM|1.0|0.100106660520033|0.103463330162708|0.0906105439583131|0.0798407800597584|0.0714483065289855|0.784502508918709|
|inceptionresnet-KonCept|relu|IFGSM|1.0|0.83149735842992|1.252678077|1.520598964|1.698436824|1.81926388|0.845597697212582|
|inceptionresnet-KonCept|silu|IFGSM|1.0|0.259084220259577|0.290124690175235|0.294807609890422|0.289799871938365|0.280418428622285|0.821794397480203|





## Links
<!-- [KonIQ-10k](https://drive.google.com/file/d/13KlUl_Uo68MDjL_ef7INQHf_waDZf4R9/view?usp=drive_link)<br>
[Checkpoints-SiLU](https://drive.google.com/file/d/19sbNdE7EJDQScCWgPuQpxgptYt1YckM0/view?usp=drive_link)<br>
[Checkpoints-ReLU](https://drive.google.com/file/d/1pte9VqUfsD3Eu0DSNQYKQpbtJep2zxEB/view?usp=drive_link)<br>
[Checkpoints-SiLU_res34](https://drive.google.com/file/d/1OrU0zi8-TWI_MkE_1OetBKG-EvQ8hGpX/view?usp=drive_link)<br>
[Checkpoints-ReLU_res34](https://drive.google.com/file/d/1kDGQ96qYbZuXXpqiT-BD-pk8TY_WNtkS/view?usp=drive_link)<br>
[Checkpoints-ReLU-SiLU_res34](https://drive.google.com/file/d/1F-9J2R9j6ID5ln-ZPPcDPzUlXZX0PuFr/view?usp=drive_link)<br> -->
