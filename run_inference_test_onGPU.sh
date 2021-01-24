#!/bin/bash

PYTHON=$(pwd)/eacl2021-env/bin/python

if [ "$1" == "" ]; then
    echo "Give file name as a program argument"
    exit 1
fi


echo "model_type,batch_size,avg_time,std_time,median_time" > $1

# test our model
for batch_size in 16 32 64 128; do
    echo RUNNING ours $batch_size
    CUDA_VISIBLE_DEVICES="0" $PYTHON our-model-inference-test.py $batch_size -o $1
done

for model_type in 'distilbert' 'bert_x12' 'bert_x24' 'albert_v2_x12' 'longformer_x12'; do
    for batch_size in 16 32 64 128; do
      echo RUNNING $model_type $batch_size
      CUDA_VISIBLE_DEVICES="0" $PYTHON transform-inference-test.py $model_type $batch_size -o $1
    done 
done

# longformer is to big to run with this batch settings so we run with lower batch sizes

echo RUNNING longformer_x12 8
CUDA_VISIBLE_DEVICES="0" $PYTHON transform-inference-test.py 'longformer_x12' 8 -o $1