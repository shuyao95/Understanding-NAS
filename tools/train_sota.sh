#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py --drop_path_prob 0.2 --auxiliary --cutout --arch NASNet_IMP --save NASNet_IMP --seed 0 --batch_size 60 --data ../data/cifar10 &
CUDA_VISIBLE_DEVICES=3 python train.py --drop_path_prob 0.2 --auxiliary --cutout --arch ENAS_IMP --save ENAS_IMP --seed 0 --batch_size 60 --data ../data/cifar10 &
CUDA_VISIBLE_DEVICES=4 python train.py --drop_path_prob 0.2 --auxiliary --cutout --arch AmoebaNet_IMP --save AmoebaNet_IMP --seed 0 --batch_size 60 --data ../data/cifar10 &
CUDA_VISIBLE_DEVICES=5  python train.py --drop_path_prob 0.2 --auxiliary --cutout --arch SNAS_IMP --save SNAS_IMP --seed 0 --batch_size 60 --data ../data/cifar10
