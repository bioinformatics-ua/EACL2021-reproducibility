#!/bin/bash

./run_inference_test_onCPU.sh csv/$1_inferenceOnCPU.csv
./run_inference_test_onGPU.sh csv/$1_inferenceOnGPU.csv

./run_train_test_onCPU.sh csv/$1_trainOnCPU.csv
./run_train_test_onGPU.sh csv/$1_trainOnGPU.csv

./run_train_test_onCPU_only_classifiers.sh csv/$1_trainOnCPU_only_classifiers.csv
./run_train_test_onGPU_only_classifiers.sh csv/$1_trainOnGPU_only_classifiers.csv