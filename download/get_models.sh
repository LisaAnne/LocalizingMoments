#!/bin/bash

mkdir -p snapshots
mkdir -p prototxts

cd prototxts

wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/models/deploy_clip_retrieval_rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt

wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/models/deploy_clip_retrieval_flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt

cd ../snapshots

wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/models/rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2_iter_30000.caffemodel

wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/models/flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2_iter_30000.caffemodel

cd ..

cd data

wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5

wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.50d.txt
rm glove.6B.100d.txt
rm glove.6B.200d.txt

cd ..
