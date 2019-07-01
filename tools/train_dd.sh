#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dd.py --drop_path_prob 0 --arch concat --save DARTS_DENSE_CONCAT --seed 0
CUDA_VISIBLE_DEVICES=0 python train_dd.py --drop_path_prob 0 --arch add --save DARTS_DENSE_add --seed 0
CUDA_VISIBLE_DEVICES=0 python train_dd.py --drop_path_prob 0 --arch nogroup --save DARTS_DENSE_NOGROUP --seed 0
CUDA_VISIBLE_DEVICES=0 python train_dd.py --drop_path_prob 0 --arch concat --save DARTS_DENSE_CONCAT_PREV-1 --seed 0 --prev_connects 1
CUDA_VISIBLE_DEVICES=0 python train_dd.py --drop_path_prob 0 --arch seq --save DARTS_DENSE_SEQ --seed 0