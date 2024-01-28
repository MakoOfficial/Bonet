#!/bin/bash
START_EPOCH=0
NUM_EPOCHS=60
LR=0.0001
PATIENCE=2
BATCH_SIZE=20
NUM_WORKERS=8
NUM_GPUS=1
GPUS=0

SSD_LOCATION="../../autodl-tmp/Bonet"
DATASET="RSNA" #RHPE
EXPERIMENT_NAME="Experiment"

DATA_TRAIN="../../autodl-tmp/archive/train"
ANN_PATH_TRAIN="../../autodl-tmp/archive/train.csv"
ROIS_PATH_TRAIN=

DATA_VAL="../../autodl-tmp/archive/valid"
ANN_PATH_VAL="../../autodl-tmp/archive/valid.csv"
ROIS_PATH_VAL=

DATA_TEST=
ANN_PATH_TEST=
ANN_PATH_TEST=

SAVE_FOLDER=$SSD_LOCATION"/TRAIN/"$EXPERIMENT_NAME
mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_snapshot.pth"
OPTIM_SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_optim.pth"


CUDA_VISIBLE_DEVICES=0 python train.py --data-train "../../autodl-tmp/archive/train" --ann-path-train "../../autodl-tmp/archive/train.csv" --data-val "../../autodl-tmp/archive/valid" --ann-path-val "../../autodl-tmp/archive/valid.csv" --batch-size 20 --start-epoch 0 --epochs 60 --lr 1e-4 --patience 2 --gpu 0 --save-folder "../../autodl-tmp/Bonet/TRAIN/Experiment" --dataset "RSNA" --eval-first --workers 8 #>> $SAVE_FOLDER"/log.txt" #Uncomment if you want a log of your training
