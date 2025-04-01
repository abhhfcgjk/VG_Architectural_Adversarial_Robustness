from pathlib import Path
import random
from typing import Any

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def, Pipeline


class ExternalInputIterator(object):
    def __init__(self, img_dir, batch_size, device_id, num_gpus, shuffle=False):
        self.images_dir = Path(img_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.files = sorted(self.images_dir.glob('*.png'), key=lambda item: item.name)

        # self.files = sorted(self.images_dir.glob('*.jpg'), key=lambda item: item.name)
        self.n = len(self.files)
        print(self.n)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            random.shuffle(self.files)
        return self

    def __next__(self):
        batch = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            if self.i == self.n:
                break
            jpeg_filename = self.files[self.i]
            batch.append(np.fromfile(jpeg_filename, dtype=np.uint8))
            self.i += 1

        return [batch]

    def __len__(self):
        return self.n


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, seed):
        super().__init__(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed
        )
        self.external_input = ops.ExternalSource(source=external_data, num_outputs=1)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)

    def define_graph(self):
        images = self.external_input()
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255, 255, 255],
        )
        images = fn.resize(images, size=(224, 224))

        return images


def get_data_loader(
    directory: str,
    rank: int,
    num_tasks: int,
    batch_size: int,
    num_workers: int,
    seed: int,
):  
    eii = ExternalInputIterator(
        img_dir=directory,
        batch_size=batch_size,
        device_id=rank,
        num_gpus=num_tasks,
    )
    test_pipe = ExternalSourcePipeline(
        batch_size=batch_size, num_threads=num_workers, device_id=rank, external_data=eii, seed=seed
    )
    test_pipe.build()
    test_loader = DALIGenericIterator(
        test_pipe,
        output_map=['data'],
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    return test_loader


if __name__ == '__main__':
    train_loader = get_data_loader(
        directory='NIPS',
        rank=0,
        num_tasks=1,
        batch_size=16,
        num_workers=4,
        seed=1,
    )

    num_samples = 0
    for step, data in enumerate(train_loader):
        inputs = data[0]['data']
        print(inputs.shape)
        num_samples += inputs.shape[0]
        break

    print(num_samples)
