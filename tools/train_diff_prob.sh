#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0.1 --arch DARTS_CONN3 --save DARTS_CONN3_PROB-0.1 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0.15 --arch DARTS_CONN3 --save DARTS_CONN3_PROB-0.15 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0.2 --arch DARTS_CONN3 --save DARTS_CONN3_PROB-0.2 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0.25 --arch DARTS_CONN3 --save DARTS_CONN3_PROB-0.25 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0.3 --arch DARTS_CONN3 --save DARTS_CONN3_PROB-0.3 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0.4 --arch DARTS_CONN3 --save DARTS_CONN3_PROB-0.4 --seed 0

