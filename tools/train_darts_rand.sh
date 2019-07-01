#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS2_CONN1 --save DARTS_OPS2_CONN1 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS2_CONN2 --save DARTS_OPS2_CONN2 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS2_CONN3 --save DARTS_OPS2_CONN3 --seed 0
CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS2_CONN4 --save DARTS_OPS2_CONN4 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS3_CONN1 --save DARTS_OPS3_CONN1 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS3_CONN2 --save DARTS_OPS3_CONN2 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS3_CONN3 --save DARTS_OPS3_CONN3 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS3_CONN4 --save DARTS_OPS3_CONN4 --seed 0
#CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS4_CONN1 --save DARTS_OPS4_CONN1 --seed 0
#CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS4_CONN2 --save DARTS_OPS4_CONN2 --seed 0
#CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS4_CONN3 --save DARTS_OPS4_CONN3 --seed 0
#CUDA_VISIBLE_DEVICES=0 python train.py --drop_path_prob 0 --arch DARTS_OPS4_CONN4 --save DARTS_OPS4_CONN4 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS5_CONN1 --save DARTS_OPS5_CONN1 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS5_CONN2 --save DARTS_OPS5_CONN2 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS5_CONN3 --save DARTS_OPS5_CONN3 --seed 0
#CUDA_VISIBLE_DEVICES=1 python train.py --drop_path_prob 0 --arch DARTS_OPS5_CONN4 --save DARTS_OPS5_CONN4 --seed 0

