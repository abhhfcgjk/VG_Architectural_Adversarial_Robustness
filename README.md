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
CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 -gr --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-3 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 -adv --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet101 -clp --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet34 --loss_type norm-in-norm --p 1 --q 2 --activation silu --pbar

CUDA_VISIBLE_DEVICES=2 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet101 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --width_prune 120 --height_prune 90 --kernel_prune 1 --pls_images 100  --pbar 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --feature_model debiased --mgamma 0.1 --debug
CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 8 -e 5 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 -arch inceptionresnet --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept -rs_h 384 -rs_w 512 --debug
CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-5 -bs 8 -e 60 --ft_lr_ratio 0.1 -arch inceptionresnet --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept -rs_h 384 -rs_w 512 --debug

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 70 --ft_lr_ratio 0.1 -arch inceptionresnet --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept -rs_h 384 -rs_w 512

CUDA_VISIBLE_DEVICES=0 python train.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-4 -bs 8 -e 70 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu --pbar --model KonCept
```
## Testing
```
python main.py -arch resnet101 --activation relu --device cuda --csv_results_dir rs --attack_type UAP -iter 10
CUDA_VISIBLE_DEVICES=1 nohup python main.py -arch resnet101 --activation relu --device cuda --csv_results_dir rs -iter 1 &
python main.py -arch resnet50 --activation Fsilu --device cuda --csv_results_dir rs -iter 1 --model KonCept
python main.py -arch resnet101 -adv --activation Fsilu --device cuda --csv_results_dir rs -iter 1
python main.py --resize -arch resnet34 --activation silu --device cuda --csv_results_dir rs -iter 10
python main.py  --resize -arch resnet50 --activation relu --device cuda --csv_results_dir rs -iter 1 -prune 0.1 -t_prune pls

CUDA_VISIBLE_DEVICES=1 nohup python main.py --resize -arch resnet101 -clp --activation relu --device cuda --csv_results_dir rs -iter 8 -prune 0.1 -t_prune l2 > p_clp8.out&
```

## Visualization
```
tensorboard --logdir=runs --port=6006
```
## Results									
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
| resnet50-Linearity              | relu      | IFGSM | 3.0 | 0.6649667797270242 | 0.6967321103036666 | 0.6320361080639375 | 0.5166999602273984 | 0.4095541556005846 | 0.9073427128458336 |
| resnet50-Linearity              | Fsilu     | IFGSM | 3.0 | 0.5912780611235123 | 0.8819720154159822 | 0.980485200048574  | 0.9683424939171296 | 0.9072291577510978 | 0.8059640341191723 |
| resnet50-Linearity              | silu      | IFGSM | 3.0 | 0.828807415018611  | 0.8943692225779339 | 0.8210258951154655 | 0.678200200197768  | 0.5449194495767545 | 0.8979679103300369 |
| resnet50-Linearity              | relu_silu | IFGSM | 3.0 | 0.7478465998922245 | 0.8339154854261513 | 0.7883818322018485 | 0.6711943889297809 | 0.5616276588997859 | 0.907032739330434  |
| resnet50-Linearity              | gelu      | IFGSM | 3.0 | 1.4056323819583914 | 1.626507501915638  | 1.5725895870560087 | 1.3711415994381375 | 1.1765436318171336 | 0.9054148805343132 |
| resnet50-Linearity              | elu       | IFGSM | 3.0 | 0.5544974845882255 | 0.5900700029172438 | 0.5582459215301648 | 0.4907852762536095 | 0.4259316112561266 | 0.908696613023034  |
| resnet50-Linearity              | Fgelu     | IFGSM | 3.0 | 1.926825130308132  | 2.790681870120941  | 3.1143008345592045 | 3.1021207142401925 | 2.987046477722373  | 0.830302889183858  |
| resnet50-Linearity              | Felu      | IFGSM | 3.0 | 0.175962689025801  | 0.1962690344484608 | 0.1955063935417814 | 0.1827940953101318 | 0.1672623780669155 | 0.8368197474309959 |
| resnet18-Linearity              | relu      | IFGSM | 3.0 | 0.9170081924388372 | 1.02636703288776   | 0.9823502269983664 | 0.8516014873858639 | 0.7377980873815202 | 0.8954013771159083 |
| resnet34-Linearity              | relu      | IFGSM | 3.0 | 0.5290196299308674 | 0.5552012115185135 | 0.5176550171329987 | 0.4435483856859614 | 0.3851533567452144 | 0.90180907612445   |
| wideresnet50-Linearity          | relu      | IFGSM | 3.0 | 1.7246825454310255 | 1.9094647570310768 | 1.7740503391457614 | 1.4765582071640797 | 1.2080881814627231 | 0.9141370047486748 |
| vonenet50-Linearity             | relu      | IFGSM | 3.0 | 0.4656457428133916 | 0.7665104787837267 | 0.8910454450245313 | 0.87722616118778   | 0.8300798578285498 | 0.8581454114877978 |
| advresnet50-Linearity           | relu      | IFGSM | 3.0 | 0.1927677126652597 | 0.3282881149418939 | 0.3980263765727761 | 0.4167393409884909 | 0.422672057595345  | 0.8544529631927879 |
| resnet50-Linearity+prune=0.1pls | relu      | IFGSM | 3.0 | 0.6210778811788464 | 0.6456736809064555 | 0.578595500348605  | 0.4674537782565922 | 0.3658327067443307 | 0.9053427252105404 |
| resnet50-Linearity+prune=0.1l1  | relu      | IFGSM | 3.0 | 0.6331776600799419 | 0.666414489476717  | 0.6054548593599082 | 0.4976433184612517 | 0.3985155389946695 | 0.9069346965938276 |
| resnet50-Linearity+prune=0.1l2  | relu      | IFGSM | 3.0 | 0.7212197964143024 | 0.7305801539885233 | 0.6385262614618573 | 0.4968527075836361 | 0.3741981086516324 | 0.9071260803983338 |
| resnet50-Linearity              | relu      | IFGSM | 5.0 | 1.0053892435785676 | 0.98793804287486   | 0.8541665498297351 | 0.6971170122002657 | 0.5720926001873433 | 0.9073427128458336 |
| resnet50-Linearity              | Fsilu     | IFGSM | 5.0 | 0.9735731252191332 | 1.266012472775847  | 1.306039553610815  | 1.22152177366444   | 1.1183714093452255 | 0.8059640341191723 |
| resnet50-Linearity              | silu      | IFGSM | 5.0 | 1.2620217376181209 | 1.258763344522356  | 1.095941759074064  | 0.9035345374535936 | 0.7458757854993034 | 0.8979679103300369 |
| resnet50-Linearity              | relu_silu | IFGSM | 5.0 | 1.1663151230153694 | 1.2010723237751104 | 1.0679330512901315 | 0.9058048733874478 | 0.776571447897455  | 0.907032739330434  |
| resnet50-Linearity              | gelu      | IFGSM | 5.0 | 2.257984659033206  | 2.4060068808484947 | 2.1876310162504304 | 1.8957500469925093 | 1.6451643715978876 | 0.9054148805343132 |
| resnet50-Linearity              | elu       | IFGSM | 5.0 | 0.7971277029859882 | 0.7992598181279064 | 0.7185197586796246 | 0.6233747641216943 | 0.5464095281216899 | 0.908696613023034  |
| resnet50-Linearity              | Fgelu     | IFGSM | 5.0 | 3.238034424304794  | 4.183024389095399  | 4.306458978636125  | 4.161840695309114  | 3.939633664640728  | 0.830302889183858  |
| resnet50-Linearity              | Felu      | IFGSM | 5.0 | 0.2135864248807063 | 0.2167048067758225 | 0.2039627501930991 | 0.1968421658775164 | 0.199498799816399  | 0.8368197474309959 |
| resnet18-Linearity              | relu      | IFGSM | 5.0 | 1.41055481110148   | 1.4562407055775517 | 1.3175963493479153 | 1.1420492499439612 | 1.011032570970563  | 0.8954013771159083 |
| resnet34-Linearity              | relu      | IFGSM | 5.0 | 0.6781554567402808 | 0.6618338267832035 | 0.5860717039749742 | 0.5194835152730154 | 0.498278960297437  | 0.90180907612445   |
| wideresnet50-Linearity          | relu      | IFGSM | 5.0 | 3.0068246207751828 | 3.0315249497822423 | 2.61312460619117   | 2.126918605946023  | 1.748785196307662  | 0.9141370047486748 |
| vonenet50-Linearity             | relu      | IFGSM | 5.0 | 0.8276130829419004 | 1.2044855759505062 | 1.2956092501193004 | 1.2547122078808457 | 1.1889550794455952 | 0.8581454114877978 |
| advresnet50-Linearity           | relu      | IFGSM | 5.0 | 0.3312005520691838 | 0.4699363890985004 | 0.5223390852448799 | 0.5383308042763296 | 0.542723036940163  | 0.8544529631927879 |
| resnet50-Linearity+prune=0.1pls | relu      | IFGSM | 5.0 | 0.9535295024497    | 0.9290187233284498 | 0.7888121761926893 | 0.6335459905625421 | 0.5116008141326691 | 0.9053427252105404 |
| resnet50-Linearity+prune=0.1l1  | relu      | IFGSM | 5.0 | 0.9533877773961822 | 0.9408591269483856 | 0.8137451742559885 | 0.6680026885454268 | 0.5533942565707943 | 0.9069346965938276 |
| resnet50-Linearity+prune=0.1l2  | relu      | IFGSM | 5.0 | 1.088904951373256  | 1.0401022438512    | 0.8705946734525232 | 0.6785828887612106 | 0.5360348790029144 | 0.9071260803983338 |





## Links
__[Checkpoints](https://drive.google.com/drive/folders/1K98OLnfLZ7Q0L0kYDiuBisiMnctvYeg1?usp=sharing)__
