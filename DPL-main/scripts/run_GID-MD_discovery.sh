#!/usr/bin bash

# divide_seed=10
python main_discover.py \
  --dataset GID-MD \
  --data_dir dataset/clinc \
  --gpus 1 \
  --seed 4 \
  --max_epochs 100 \
  --batch_size 512 \
  --num_labeled_classes 90 \
  --num_unlabeled_classes 60 \
  --divide_seed 10 \
  --precision 16 \
  --proto_m 0.9 \
  --base_lr 0.1

# divide_seed=20
#python main_discover.py \
#  --dataset GID-MD \
#  --data_dir dataset/clinc \
#  --gpus 1 \
#  --seed 3 \
#  --max_epochs 100 \
#  --batch_size 512 \
#  --num_labeled_classes 90 \
#  --num_unlabeled_classes 60 \
#  --divide_seed 20 \
#  --precision 16 \
#  --proto_m 0.9 \
#  --base_lr 0.1
#
## divide_seed=30
#python main_discover.py \
#  --dataset GID-MD \
#  --data_dir dataset/clinc \
#  --gpus 1 \
#  --seed 3 \
#  --max_epochs 100 \
#  --batch_size 512 \
#  --num_labeled_classes 90 \
#  --num_unlabeled_classes 60 \
#  --divide_seed 30 \
#  --precision 16 \
#  --proto_m 0.9 \
#  --base_lr 0.1