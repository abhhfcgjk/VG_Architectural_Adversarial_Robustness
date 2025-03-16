from pathlib import Path
import random
from typing import Any

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
import pandas as pd
from torch.utils.data import DataLoader

class ExternalInputIterator(object):
    def __init__(self, batch_size, device_id, num_gpus, img_dir, data_info, phase):
        self.images_dir = Path(img_dir)
        self.batch_size = batch_size

        self.data_info = pd.read_csv(data_info)
        self.data_info = self.data_info[self.data_info["set"] == phase]
        self.data_info["MOS"] = self.data_info["MOS"].astype("float32")

        self.shuffle = phase == 'train'
        # whole data set size
        self.n = len(self.data_info)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            self.data_info.sample()
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            if self.i == self.n:
                break
            jpeg_filename = self.data_info.iloc[self.i % self.n]['image_name']
            label = self.data_info.iloc[self.i % self.n]['MOS']
            # batch.append(jpeg_filename)
            batch.append(np.fromfile(self.images_dir / jpeg_filename, dtype=np.uint8)) 
            labels.append(np.float32([label]))
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__


def ExternalSourcePipeline(batch_size, size, num_threads, device_id, external_data, seed, phase='train'):
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed)
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        if phase == 'train':
            mirror = fn.random.coin_flip(probability=0.7)
        else:
            mirror = 0
        images = fn.crop_mirror_normalize(
            images.gpu(),
            crop=size,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0., 0., 0.],
            std=[255, 255, 255],
            mirror=mirror
        )
        labels.gpu()
        pipe.set_outputs(images, labels)
    return pipe


def get_data_loaders(
    rank: int,
    num_tasks: int,
    batch_size: int,
    size: tuple[int, int],
    num_workers: int,
    seed: int,
    args: dict[str, Any],
    phase: str = "train",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare the train-val-test data.

    :param args: related arguments
    :return: train_loader, val_loader, test_loader
    """
    if phase == "train":
        eii = ExternalInputIterator(
            batch_size, 0, 1,
            img_dir=args['directory'],
            data_info=args['data_info'],
            phase='train')
        train_pipe = ExternalSourcePipeline(
            batch_size=batch_size,
            size=size,
            num_threads=num_workers,
            device_id=rank,
            external_data=eii,
            seed=seed,
            phase='train')
        train_pipe.build()
        train_loader = DALIClassificationIterator(
            train_pipe,
            last_batch_padded=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

        if rank:
            return train_loader, None, None

        eii = ExternalInputIterator(
            batch_size, 0, 1,
            img_dir=args['directory'],
            data_info=args['data_info'],
            phase='val')
        val_pipe = ExternalSourcePipeline(
            batch_size=batch_size,
            size=size,
            num_threads=num_workers,
            device_id=rank,
            external_data=eii,
            seed=seed,
            phase='val')
        val_pipe.build()
        val_loader = DALIClassificationIterator(
            val_pipe,
            last_batch_padded=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    eii = ExternalInputIterator(
        batch_size, 0, 1,
        img_dir=args['directory'],
        data_info=args['data_info'],
        phase='test')
    test_pipe = ExternalSourcePipeline(
        batch_size=batch_size,
        size=size,
        num_threads=num_workers,
        device_id=rank,
        external_data=eii,
        seed=seed,
        phase='test')
    test_pipe.build()
    test_loader = DALIClassificationIterator(
        test_pipe,
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL)

    if phase == "test":
        return test_loader

    return train_loader, val_loader, test_loader
