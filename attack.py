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

PATH = "LinearityIQA/checkpoints/p1q2.pth"
EPS = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"


class MetricModel(torch.nn.Module):
    def __init__(self, dev, model_path=PATH, batch_size=8):
        global device
        super().__init__()
        device = dev
        self.model = IQAModel(
            arch=args.architecture,
            pool=args.pool,
            use_bn_end=args.use_bn_end,
            P6=args.P6,
            P7=args.P7,
        ).to(device)
        self.lower_better = False
        self.full_reference = False

    def forward(self, image, inference=False):
        return self.model.predict_with_grads(image)[:, 0]


def test_attack(
    attack_callback,
    model,
    dataset_path="./NIPS_test/",
    checkpoints_path="LinearityIQA/checkpoints/p1q2.pth",
    attack_type="FGSM",
    device_="cpu",
    csv_results_dir=".",
):
    epsilons = np.array([2, 4, 8, 10]) / 255.0
    device = torch.device(device_)
    results = pd.DataFrame(columns=["image_name", "clear_val", "attacked_val"])
    checkpoint = torch.load(checkpoints_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # count = 5
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

        for eps in epsilons:
            im_attacked = attack_callback(
                im, model=model, metric_range=100, device=device_, eps=eps
            )

            with torch.no_grad():
                diff = im_attacked - im
                diff = torch.clamp(diff, min=-eps, max=eps)
                # print("DIFF:", torch.sum(diff))
                im_attacked = im + diff

                attacked_val = model(im_attacked)

                results.loc[len(results.index)] = [
                    Path(image_path).stem,
                    np.mean(clear_val),
                    np.mean(attacked_val),
                ]
                diffs[int(eps * 255)].append(
                    float(torch.abs(diff).mean().detach().cpu().item())
                )
        # count-=1
        # if not count:
        #     break
    res = []
    for int_eps in diffs.keys():
        res.append(np.array(diffs[int_eps]).mean())
        print(
            f"eps={int_eps}/255 :",
            "mean diff = {:.5f}".format(np.array(diffs[int_eps]).mean()),
        )

    if csv_results_dir is not None:
        csv_path = os.path.join(csv_results_dir, "results.csv")
        results.to_csv(csv_path)
        print(f"Results saved to {csv_path}")

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
        "--architecture",
        default="resnext101_32x8d",
        type=str,
        help="arch name (default: resnext101_32x8d)",
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
        default="checkpoints/p1q2.pth",
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

    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--dataset_path", type=str, default="./NIPS_test/")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available()

    model = IQAModel(
        arch=args.architecture,
        pool=args.pool,
        use_bn_end=args.use_bn_end,
        P6=args.P6,
        P7=args.P7,
    ).to(args.device)

    metric_range = 100

    total_score = test_attack(iterative.attack, model=model)
    print(
        "Result for {} type attack: {:.4f}".format(
            args.attack_type.capitalize(), total_score
        )
    )
