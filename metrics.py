import torch
from scipy import stats
import numpy as np
# from skimage.metrics import structural_similarity

from torch.utils.tensorboard import SummaryWriter


class IQAPerformance(object):
    idx_col = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.targs = []
        self.preds = []

    # only compute on image score
    def update(self, preds, targs):
        self.preds.extend([t.item() for t in preds])
        self.targs.extend([t.item() for t in targs])

    def _compute(self, func):
        def get_column(x):
            return np.reshape(np.asarray(x), (-1,))

        return func(get_column(self.targs), get_column(self.preds))[0]

    @property
    def srcc(self):
        return self._compute(stats.spearmanr)

    @property
    def plcc(self):
        return self._compute(stats.pearsonr)

    @property
    def krocc(self):
        return self._compute(stats.kendalltau)


def dump_scalar_metrics(
    metrics: dict,
    writer: SummaryWriter,
    phase: str,
    global_step: int = 0,
    dataset: str = "",
    **kwargs
):
    prefix = phase + (f"_{dataset}" if dataset else "")
    wandb = kwargs.get("wandb")
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f"{metric_name}/{prefix}",
            metric_value,
            global_step=global_step,
        )
        wandb.log({f"{metric_name}/{prefix}": metric_value})


def add_metrics_dict(metrics: dict, metrics_new: dict) -> dict:
    for key, value in metrics_new.items():
        metrics[key] = metrics.get(key, 0) + value
    return metrics


def divide_metrics(metrics: dict, n: int) -> dict:
    return {key: (value / n) for key, value in metrics.items()}


# def ssim_metric(x: torch.tensor, y: torch.tensor):
#     return structural_similarity(
#         x.cpu().detach().permute(0, 2, 3, 1).numpy()[0],
#         y.cpu().detach().permute(0, 2, 3, 1).numpy()[0],
#         channel_axis=2,
#         data_range=1,
#     )


# def compute_ssim(x: torch.tensor, y: torch.tensor):
#     ssim = [
#         ssim_metric(img1.unsqueeze(0), img2.unsqueeze(0)) for img1, img2 in zip(x, y)
#     ]
#     return torch.tensor(ssim, device=x.device)
