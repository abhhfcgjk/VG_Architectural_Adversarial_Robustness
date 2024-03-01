#!/bin/bash

test_all(){
    source config.conf
    local se_str=""
    local command_args=""
    for arch in "${architectures[@]}"
    do
        for activ in "${activations[@]}"
        do
            se_str=""
            for se_block in "${se[@]}"
            do
                if $se_block
                then
                    se_str="-se"
                fi

                if [ $debug ]
                then
                    iters=1
                fi
                command_args="--dataset_path $default_dataset_path --resize -arch $arch
                --activation $activ --device $device --csv_results_dir $results_dir 
                -iter ${iters} $se_str"
                echo "CONFIG: architecture=$arch activation=$activ \
results_dir=$results_dir se=${se_block}${se_str} device=$device $debug"
                python main.py ${command_args} -${debug} 2>>test_error.log
            done
        done
    done
}

train_all(){
    source config.conf
    cd LinearityIQA
    local command_args=""
    for arch in "${architectures[@]}"
    do
        for activ in "${activations[@]}"
        do
            se_str=""
            for se_block in "${se[@]}"
            do
                if $se_block
                then
                    se_str="-se"
                fi

                command_args="--dataset KonIQ-10k 
                --resize --exp_id 0 -lr 1e-4 -bs 4 -e 30 
                --ft_lr_ratio 0.1 -arch $arch 
                --loss_type norm-in-norm --p 1 --q 2 
                --activation $activ $se_str --pbar"
                echo "CONFIG: $arch $activ se=${se_block}${se_str} device=$device $debug"
                CUDA_VISIBLE_DEVICES=0 python main.py ${command_args} ${debug} 2>>train_error.log
            done
        done
    done
}


get_device(){
    is_cuda=$(python -c "from torch.cuda import is_available; print(is_available())")
    # echo $is_cuda
    if [ "$is_cuda" = "True" ]
    then
        device="cuda"
    else
        device="cpu"
    fi
    # echo "device=$device"
}

default_dataset_path="./NIPS_test/"
results_dir="rs"
debug=""
train=false
test=false
# iters=1


while getopts ":Ttdi:" keys
do
    case $keys in
        t) test=true;;
        d) debug="--debug";;
        T) train=true;;
        i) iters="$OPTARG";;
        *) echo "INVALID ARGUMENTS"
    esac
done
echo "SCRIPT STARTED"
get_device

if $test
then
    test_all
fi
if $train
then
    train_all
fi

echo "SCRIPT FINISHED"