#!/bin/bash

PYTHON=$(pwd)/eacl2021-env/bin/python

if [ "$1" == "" ]; then
    echo "Give file name as a program argument"
    exit 1
fi

echo "model_type,batch_size,avg_time,std_time,median_time" > $1

# test the transformer models
for model_type in 'distilbert' 'albert_v2_x12' 'bert_x12' 'bert_x24'; do
    for batch_size in 8 16 32; do
      echo RUNNING $model_type $batch_size
      CUDA_VISIBLE_DEVICES="0" $PYTHON transform-train-test.py $model_type $batch_size -o $1 -classifier 
    done
done


