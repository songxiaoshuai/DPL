#!/usr/bin bash

# clinc
python main_pretrain.py \
    --dataset GID-MD \
    --data_dir dataset/clinc \
    --gpus 1 \
    --divide_seed 10 \
    --precision 16 \
    --max_epochs 200 \
    --batch_size 256 \
    --num_labeled_classes 90 \
    --num_unlabeled_classes 60 \



