#!/usr/bin bash

# divide_seed=10
python main_pretrain.py \
  --dataset GID-SD \
  --data_dir dataset/banking \
  --gpus 1 \
  --max_epochs 100 \
  --batch_size 512 \
  --num_labeled_classes 46 \
  --num_unlabeled_classes 31 \
  --divide_seed 10 \
