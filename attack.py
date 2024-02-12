from torch.autograd import Variable
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from LinearityIQA.IQAmodel import IQAModel
from torchvision.transforms.functional import resize, to_tensor, normalize
import iterative

from LinearityIQA.activ import ReLU_to_SILU, ReLU_to_ReLUSiLU


PATH = "LinearityIQA/checkpoints/p1q2.pth"
# PATH_silu = "LinearityIQA/checkpoints/activation=silu-resnext101_32x8d-avg-bn_end=False-loss=norm-in-norm-p=1.0-q=2.0-detach-False-ft_lr_ratio=0.1-alpha=[1, 0]-beta=[0.1, 0.1, 1]-KonIQ-10k-res=True-256x256-aug=False-monotonicity=False-lr=0.0001-bs=4-e=30-opt_level=O1-EXP0"
# PATH_relu = "LinearityIQA/checkpoints/activation=relu-resnext101_32x8d-avg-bn_end=False-loss=norm-in-norm-p=1.0-q=2.0-detach-False-ft_lr_ratio=0.1-alpha=[1, 0]-beta=[0.1, 0.1, 1]-KonIQ-10k-res=True-256x256-aug=False-monotonicity=False-lr=0.0001-bs=4-e=30-opt_level=O1-EXP0"
# PATH_relu = "LinearityIQA/checkpoints/activation=relu-resnext101_32x8d-avg-bn_end=False-loss=norm-in-norm-p=1.0-q=2.0-detach-False-ft_lr_ratio=0.1-alpha=[1, 0]-beta=[0.1, 0.1, 1]-KonIQ-10k-res=True-256x256-aug=False-monotonicity=False-lr=0.0001-bs=4-e=30-opt_level=O1-EXP0"
PATH_silu       = "LinearityIQA/checkpoints/activation=silu-resnet34-avg-bn_end=False-loss=norm-in-norm-p=1.0-q=2.0-detach-False-ft_lr_ratio=0.1-alpha=[1, 0]-beta=[0.1, 0.1, 1]-KonIQ-10k-res=True-498x664-aug=False-monotonicity=False-lr=0.0001-bs=8-e=30-opt_level=O1-EXP0"
PATH_relu       = "LinearityIQA/checkpoints/activation=relu-resnet34-avg-bn_end=False-loss=norm-in-norm-p=1.0-q=2.0-detach-False-ft_lr_ratio=0.1-alpha=[1, 0]-beta=[0.1, 0.1, 1]-KonIQ-10k-res=True-498x664-aug=False-monotonicity=False-lr=0.0001-bs=8-e=30-opt_level=O1-EXP0"
PATH_relu_silu  = "LinearityIQA/checkpoints/activation=relu_silu-resnet34-avg-bn_end=False-loss=norm-in-norm-p=1.0-q=2.0-detach-False-ft_lr_ratio=0.1-alpha=[1, 0]-beta=[0.1, 0.1, 1]-KonIQ-10k-res=True-498x664-aug=False-monotonicity=False-lr=.0001-bs=8-e=30-opt_level=O1-EXP0"

EPS = 1e-6
# device = "cuda" if torch.cuda.is_available() else "cpu"



def test_attack(
    attack_callback,
    model,
    dataset_path="./NIPS_test/",
    activation="relu",
    device_="cpu",
    csv_results_dir=".",
    debug=False
):
    checkpoints_path = PATH
    if activation=='silu':
        checkpoints_path = PATH_silu
    elif activation=='relu':
        checkpoints_path = PATH_relu
    elif activation=='relu_silu':
        checkpoints_path = PATH_relu_silu

    epsilons = np.array([2, 4, 6, 8, 10]) / 255.0
    # iterations = np.array([1, 2, 3, 4, 5])
    device = torch.device(device_)
    results = pd.DataFrame(columns=["image_name", "clear_val", "attacked_val"])
    data = [pd.DataFrame(columns=["image_num", "clear_val", "attacked_val"]) for _ in range(len(epsilons))]
    checkpoint = torch.load(checkpoints_path, map_location=device_)
    # print(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    k = checkpoint['k']
    b = checkpoint['b']
    max_pred = checkpoint['max']
    min_pred = checkpoint['min']
    print(min_pred, max_pred)

    count = 5
    image_num = 0
    for image_path in tqdm(
        Path(dataset_path).iterdir(),
        total=len([x for x in Path(dataset_path).iterdir()]),
    ):
        if Path(image_path).suffix not in [".png", ".jpg", ".jpeg"]:
            continue
        diffs = {int(x * 255): [] for x in epsilons}
        im = Image.open(image_path).convert("RGB")
        if args.resize:  # resize or not?
            im = resize(im, (args.resize_size_h, args.resize_size_w))  #
        im = to_tensor(im).to(device)

        im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        im = im.unsqueeze(0)

        with torch.no_grad():
            clear_val = model(im)
            clear_val = clear_val[-1].item()*k[0] + b[0]
            print(clear_val, k[0], b[0])
            # mean_clear_val = np.mean([elem.cpu().numpy() for elem in clear_val])
            # print(clear_val, mean_clear_val)


        for eps_i, eps in enumerate(epsilons):
            im_attacked = attack_callback(
                im, model=model, metric_range=100, device=device_, 
                eps=10/255, iters=1, alpha=eps, k=k, b=b, mmin=min_pred, mmax=max_pred
            )

            with torch.no_grad():
                diff = im_attacked - im
                diff = torch.clamp(diff, min=-eps, max=eps)
                # print("DIFF:", torch.sum(diff))
                im_attacked = im + diff

                attacked_val = model(im_attacked) # поменять значение
                attacked_val = attacked_val[-1].item()*k[0] + b[0]
                # mean_attacked_val = np.mean([elem.cpu().numpy() for elem in attacked_val])

                results.loc[len(results.index)] = [
                    Path(image_path).stem,
                    clear_val,
                    attacked_val,
                ]
                diffs[int(eps * 255)].append(
                    float(torch.abs(diff).mean().detach().cpu().item())
                )
                data[eps_i].loc[len(data[eps_i].index)] = [
                    image_num,
                    clear_val,
                    attacked_val,
                ]
        if debug:
            count-=1
            if not count:
                break
        image_num += 1
    res = []
    for int_eps in diffs.keys():
        res.append(np.array(diffs[int_eps]).mean())
        print(
            f"eps={int_eps}/255 :",
            "mean diff = {:.5f}".format(np.array(diffs[int_eps]).mean()),
        )

    if csv_results_dir is not None:
        csv_path = os.path.join(csv_results_dir, "results_{}.csv".format(activation))
        results.to_csv(csv_path)
        print(f"Results saved to {csv_path}")

    if not os.path.exists('graph_data'):
        os.makedirs('graph_data')
    for i, result in enumerate(data):
        csv_data_path = f"graph_data/eps={np.int32(epsilons[i]*255)}-activation={activation}.csv"
        result.to_csv(csv_data_path)
        print(f"{csv_data_path} saved.")

    return np.array(res).mean()




if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--attack_type", type=str, default="iterative")
    # # parser.add_argument("--uap_train_path", type=str, default="../uap_trained_data/pretrained_uap_paq2piq.png")
    # # parser.add_argument("--csv_results_dir", type=str, default=None)
    # # parser.add_argument("--model_weights", type=str, default='../weights/RoIPoolModel.pth')
    # # parser.add_argument("--dataset_path", type=str, default='../NIPS_test/')
    # parser.add_argument("--device", type=str, default='cpu') # cuda:0

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
        default=PATH,
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

    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--dataset_path", type=str, default="./NIPS_test/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--csv_results_dir", type=str, default=".")
    parser.add_argument("--debug", action="store_true")

    # parser.add_argument('--trained_model_file', default='LinearityIQA/checkpoints/p1q2.pth', type=str,
    #                 help='trained_model_file')

    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available()

    print(args.architecture)
    model = IQAModel(
        arch=args.architecture,
        pool=args.pool,
        use_bn_end=args.use_bn_end,
        P6=args.P6,
        P7=args.P7,
    ).to(args.device)

    if args.activation.lower() == 'silu':
        ReLU_to_SILU(model)
    elif args.activation.lower() == 'relu_silu':
        ReLU_to_ReLUSiLU(model)

    metric_range = 100



    print("Device: ", args.device)
    total_score = test_attack(iterative.attack if args.attack_type=="FGSM" else None, model=model, 
                              activation=args.activation, device_=args.device,
                              csv_results_dir=args.csv_results_dir, debug=args.debug)
    print(
        "Result for {} type attack: {:.4f}".format(
            args.attack_type.capitalize(), total_score
        )
    )
