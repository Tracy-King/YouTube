#!/bin/bash

#$-l rt_F=1
#$-l h_rt=12:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source ~/ytb/bin/activate
module load python/3.8.3 cuda/11.2.146 cudnn/8.1 gcc/10.1.0-cuda 
#python3 train_self_supervised.py -d 1kxCz6tt2MU_v3.10  --dataset_r1 0.70 --dataset_r2 0.85 --prefix tgn-attn-1kxCz6tt2MU_v3.10  --n_runs 1 --label superchat
#python3 train_supervised.py -d concat --n_decoder 50 --dataset_r1 0.75 --dataset_r2 0.75 --prefix tgn-attn-concat --n_runs 1 --label superchat
#python3 train_supervised.py -d concat --n_decoder 50 --dataset_r1 0.80 --dataset_r2 0.80 --prefix tgn-attn-concat --n_runs 1 --label superchat
#python3 train_supervised.py -d concat_v2 --n_epoch 5 --n_decoder 50 --dataset_r1 0.85 --dataset_r2 0.85 --prefix tgn-attn-concat --n_runs 1 --label superchat
#python3 train_supervised.py -d concat_v2 --n_epoch 5 --n_decoder 100 --dataset_r1 0.95 --dataset_r2 0.95 --prefix tgn-attn-concat_v2 --n_runs 1 --label superchat
python3 train_supervised_gat.py -d concat_half_v3.10 --n_epoch 10 --bs 500 --dataset_r1 0.90 --dataset_r2 0.95 --prefix tgn-attn-concat_half_v3.10  --n_runs 1
#python3 train_supervised.py -d 1kxCz6tt2MU_v3.10 --bs 5000 --n_epoch 5 --n_decoder 20 --n_head 2 --dataset_r1 0.90 --dataset_r2 0.95 --prefix tgn-attn-1kxCz6tt2MU_v3.10 --n_runs 1 --label superchat
