import torch
from scipy import stats
import numpy as np
# from skimage.metrics import structural_similarity

from torch.utils.tensorboard import SummaryWriter


class IQAPerformance(object):
    idx_col = 0

    def __init__(self, num_patches):
        self.num_patches = num_patches
        self.reset()

    def reset(self):
        self.targs = []
        self.preds = []

    # only compute on image score
    def update(self, preds, targs):
        self.preds.extend([t.item() for t in preds])
        self.targs.extend([t.item() for t in targs])

    def _compute(self, func):
        pred_scores = np.mean(np.reshape(np.array(self.preds), (-1, self.num_patches)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(self.targs), (-1, self.num_patches)), axis=1)
        def get_column(x):
            return np.reshape(np.asarray(x), (-1,))
        return func(get_column(gt_scores), get_column(pred_scores))[0]

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
    writer,
    phase: str,
    global_step: int = 0,
    dataset: str = "",
    **kwargs
):
    prefix = phase + (f"_{dataset}" if dataset else "")
    for metric_name, metric_value in metrics.items():
        writer.log({f"{metric_name}/{prefix}": metric_value})


def add_metrics_dict(metrics: dict, metrics_new: dict) -> dict:
    for key, value in metrics_new.items():
        metrics[key] = metrics.get(key, 0) + value
    return metrics


def divide_metrics(metrics: dict, n: int) -> dict:
    return {key: (value / n) for key, value in metrics.items()}
