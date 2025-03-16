from pathlib import Path
import random
from typing import Any, Tuple
import os

import cv2
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
import pandas as pd
from torch.utils.data import DataLoader


class ExternalInputIterator(object):
    def __init__(self, batch_size, device_id, num_gpus, patch_size, num_patches,
                 img_dir, data_info, patch_dir, patch_info, phase, prepair_patches=False):
        self.images_dir = Path(img_dir)
        self.batch_size = batch_size
        self.patch_dir = Path(patch_dir)
        self.patch_dir.mkdir(parents=True, exist_ok=True)

        self.data_info = pd.read_csv(data_info)
        self.data_info = self.data_info[self.data_info["set"] == phase]
        self.data_info["MOS"] = self.data_info["MOS"].astype("float32")

        self.shuffle = (phase == 'train')
        print(phase)

        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.patch_info = patch_info
        # if not os.path.exists(patch_info):
        if prepair_patches:
            self._prepare_patches(patch_info)
        # else:
        #     print(f"{patch_info} already exists")

        self.patch_info = pd.read_csv(patch_info)
        self.patch_files = list(self.patch_info["patch_name"])
        self.n = len(self.patch_files)
        print(self.n)

    def _prepare_patches(self, patch_info):
        """ Generate patches from images and save them if not already present. """
        print('Patch prepairing')
        patch_records = []
        for idx, row in self.data_info.iterrows():
            img_path = self.images_dir / row['image_name']
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w, _ = img.shape
            base_name = row['image_name'].split('.')[0]  # Remove file extension

            for i in range(self.num_patches):
                # Random crop
                x = random.randint(0, w - self.patch_size[0])
                y = random.randint(0, h - self.patch_size[1])
                patch = img[y:y+self.patch_size[0], x:x+self.patch_size[1]]

                # Save patch
                patch_filename = self.patch_dir / f"{base_name}_patch{i}.jpg"
                if not patch_filename.exists():
                    cv2.imwrite(str(patch_filename), patch)
                patch_records.append({
                    "patch_name": patch_filename.name,
                    "original_image": row["image_name"],
                    "MOS": row["MOS"],
                    "set": row["set"]
                })

        if os.path.exists(patch_info):
            tmp = pd.read_csv(patch_info)
            patch_df = pd.concat([tmp, pd.DataFrame(patch_records)])
            print('Readed', tmp['set'].values[0])
        else:
            patch_df = pd.DataFrame(patch_records)
        patch_df.to_csv(patch_info, index=False)
        print(f'Patch info saved to {patch_info}')

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            self.patch_info.sample()
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            # if self.i == self.n:
            #     break
            jpeg_filename = self.patch_info.iloc[self.i % self.n]['patch_name']
            label = self.patch_info.iloc[self.i % self.n]['MOS']
            b = np.fromfile(self.patch_dir / jpeg_filename, dtype=np.uint8)
            batch.append(b) 
            labels.append(np.float32([label]))
            self.i += 1
        # assert len(batch)==self.batch_size
        return (batch, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__


def ExternalSourcePipeline(batch_size, num_threads, 
                           device_id, external_data, seed, phase='train'):
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed)
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        if phase == 'train':
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            mirror = 0
        images = fn.crop_mirror_normalize(
            images.gpu(),
            # crop=(224, 224),
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
    num_patches: int,
    patch_size: Tuple[int, int],
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
            patch_size, num_patches,
            img_dir=args['directory'],
            patch_dir=args['patch_dir'],
            data_info=args['data_info'],
            patch_info=args['patch_info'],
            phase='train')
        train_pipe = ExternalSourcePipeline(
            batch_size=batch_size,
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
            patch_size, num_patches,
            img_dir=args['directory'],
            patch_dir=args['patch_dir'],
            data_info=args['data_info'],
            patch_info=args['patch_info'],
            phase='val')
        val_pipe = ExternalSourcePipeline(
            batch_size=batch_size,
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
        patch_size, num_patches,
        img_dir=args['directory'],
        patch_dir=args['patch_dir'],
        data_info=args['data_info'],
        patch_info=args['patch_info'],
        phase='test')
    test_pipe = ExternalSourcePipeline(
        batch_size=batch_size,
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
