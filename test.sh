#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python test.py --cayley3 --iters 1
CUDA_VISIBLE_DEVICES=1 python test.py --cayley3 --iters 3
CUDA_VISIBLE_DEVICES=1 python test.py --cayley3 --iters 5
CUDA_VISIBLE_DEVICES=1 python test.py --cayley3 --iters 8