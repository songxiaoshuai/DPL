#!/usr/bin bash

# divide_seed=10
python main_discover.py \
  --dataset GID-SD \
  --data_dir dataset/banking \
  --gpus 1 \
  --seed 5 \
  --max_epochs 100 \
  --batch_size 512 \
  --num_labeled_classes 46 \
  --num_unlabeled_classes 31 \
  --divide_seed 10 \
  --precision 16 \
  --proto_m 0.9 \
  --base_lr 0.02

# divide_seed=20
python main_discover.py \
  --dataset GID-SD \
  --data_dir dataset/banking \
  --gpus 1 \
  --seed 2 \
  --max_epochs 100 \
  --batch_size 512 \
  --num_labeled_classes 46 \
  --num_unlabeled_classes 31 \
  --divide_seed 20 \
  --precision 16 \
  --proto_m 0.9 \
  --base_lr 0.02

# divide_seed=30
python main_discover.py \
  --dataset GID-SD \
  --data_dir dataset/banking \
  --gpus 1 \
  --seed 0 \
  --max_epochs 100 \
  --batch_size 512 \
  --num_labeled_classes 46 \
  --num_unlabeled_classes 31 \
  --divide_seed 30 \
  --precision 16 \
  --proto_m 0.9 \
  --base_lr 0.02