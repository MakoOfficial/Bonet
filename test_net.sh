#!/bin/bash
START_EPOCH=0
NUM_EPOCHS=20
LR=0.0001
PATIENCE=2
BATCH_SIZE=1
NUM_WORKERS=0
NUM_GPUS=1
GPUS=0

SSD_LOCATION="../models/modelsRecord/Bonet/"
DATASET="RSNA" #"RHPE"
EXPERIMENT_NAME="Experiment"
DATA_TEST="../archiveMasked/maskAll/archive/valid"
ANN_PATH_TEST="../archiveMasked/maskAll/archive/valid.csv"
ROIS_PATH_TEST=None
SAVE_FOLDER=$SSD_LOCATION"/TRAIN/"$EXPERIMENT_NAME
SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_snapshot.pth"



#CUDA_VISIBLE_DEVICES=$GPUS mpirun -np $NUM_GPUS -H localhost:$NUM_GPUS -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_CUDA_HOME=/usr/local/cuda-10.0 -mca pml ob1 -mca btl ^openib python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST --rois-path-test $ROIS_PATH_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --snapshot $SNAPSHOT --cropped --dataset $DATASET
python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST --rois-path-test $ROIS_PATH_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS