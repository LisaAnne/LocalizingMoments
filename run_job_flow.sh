#!/bin/bash

lw=0.2

#check if files have been copied to ssd

didemo_data=/mnt/ssd/tmp/lhendric/didemo_data/
kinetics_features_ssd=/mnt/ssd/tmp/lhendric/kinetics_data

mkdir -p /mnt/ssd/tmp/lhendric/kinetics_data
mkdir -p $didemo_data 
mkdir -p /mnt/ssd/tmp/lhendric/snapshots

if [ ! -d $kinetics_features_ssd ]; then
    cp -r /mnt/ilcompf2d0/data/didemo_lhendric/kinetics_features/ $kinetics_features_ssd 
fi

if [ ! -f '/mnt/tmp/ssd/lhendric/didemo_data/average_fc7_feats_fps.h5' ]; then
    cp data/average_fc7_feats_fps.h5 /mnt/ssd/tmp/lhendric/didemo_data/ 
fi


python build_net.py --feature_process_visual feature_process_context  \
                    --loc \
                    --vision_layers 2 \
                    --dropout_visual 0.3 \
                    --dropout_language 0.0 \
                    --language_layers lstm_no_embed \
                    --feature_process_language recurrent_embedding \
                    --visual_embedding_dim 500 100 \
                    --language_embedding_dim 1000 100 \
                    --gpu 0 \
                    --max_iter 30000 \
                    --snapshot 10000 \
                    --stepsize 10000 \
                    --base_lr 0.05 \
                    --train_h5 /home/lisaanne/projects/LocalizingMoments/data/average_global_flow.h5 \
                    --test_h5 /home/lisaanne/projects/LocalizingMoments/data/average_global_flow.h5 \
                    --train_json data/train_data.json \
                    --test_json data/val_data.json \
                    --random_seed 1701 \
                    --loss_type triplet \
                    --lw_inter 0.2 \
                    --tag flow_hachiko_
