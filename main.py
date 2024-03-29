# from torch.autograd import Variable
# import torch
# import os
# import pandas as pd
import numpy as np
# from PIL import Image
# from pathlib import Path
# from tqdm import tqdm
# from scipy import stats
import argparse
# from LinearityIQA.IQAmodel import IQAModel
import LinearityIQA
from torchvision.transforms.functional import resize, to_tensor, normalize
# print(__package__)
# import iterative
from attack_cls import Attack
# from LinearityIQA.activ import ReLU_to_SILU, ReLU_to_ReLUSiLU


EPS = 1e-6
# device = "cuda" if torch.cuda.is_available() else "cpu"



def run(args):
    exec = Attack(LinearityIQA.IQAModel, 
                  arch=args.architecture, pool=args.pool,
                  use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7,
                  activation=args.activation, se=args.squeeze_excitation,
                  device=args.device
                  )
    # print(exec.model.state_dict().keys())
    # quit()

    exec.load_checkpoints(checkpoints_path=args.trained_model_file)
    exec.set_load_conf(dataset_path=args.dataset_path,
                       resize_size_h=args.resize_size_h, 
                       resize_size_w=args.resize_size_w)
    
    exec.attack(attack_type=args.attack_type,
                iterations=args.iterations, debug=args.debug)
    
    exec.save_results(args.csv_results_dir)

    return np.array(exec.res).mean()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Demo for LinearityIQA")

    parser.add_argument(
        "--architecture", "-arch",
        default="resnet34",
        type=str,
        help="arch name (default: resnet34)",
    )
    parser.add_argument(
        "--pool", default="avg", type=str, help="pool method (default: avg)"
    )
    parser.add_argument(
        "--use_bn_end", action="store_true", help="Use bn at the end of the output?"
    )
    parser.add_argument("--P6", type=int, default=1, help="P6 (default: 1)")
    parser.add_argument("--P7", type=int, default=1, help="P7 (default: 1)")

    parser.add_argument(
        "--trained_model_file",
        default="LinearityIQA/checkpoints/p1q2.pth",
        type=str,
        help="trained_model_file",
    )

    parser.add_argument("--resize", action="store_true", help="Resize?")
    parser.add_argument(
        "--resize_size_h", default=498, type=int, help="resize_h (default: 498)"
    )
    parser.add_argument(
        "--resize_size_w", default=664, type=int, help="resize_w (default: 664)"
    )

    parser.add_argument("-iter", "--iterations", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--attack_type", type=str, default="IFGSM")
    parser.add_argument("--dataset_path", type=str, default="./NIPS_test/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--csv_results_dir", type=str, default=".")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-se", "--squeeze_excitation", action="store_true")
    parser.add_argument("-weights", "--checkpoints_dir", type=str, default='LinearityIQA/checkpoints')

    args = parser.parse_args()


    print(args.architecture)

    path = '{}/activation={}-{}-loss=norm-in-norm-p=1.0-q=2.0-detach-False-KonIQ-10k-res={}-{}x{}-se={}'.format(args.checkpoints_dir,
                                                                                                                args.activation, 
                                                                                                                args.architecture, 
                                                                                                                args.resize, 
                                                                                                                args.resize_size_h,
                                                                                                                args.resize_size_w,
                                                                                                                args.squeeze_excitation)

    print("Device: ", args.device)
    print(path)
    args.trained_model_file = path
    total_score = run(args)
    print(
        "Result for {} type attack: {:.4f}".format(
            args.attack_type.capitalize(), total_score
        )
    )
